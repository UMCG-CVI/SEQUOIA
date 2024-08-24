"""
SEQUOIA - automated multiclass SEgmentation, QUantification, and visualizatiOn of the dIseased Aorta on hybrid PET/CT


Copyright 2023 GD van Praagh

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import datetime
import preprocessing_augmentation as pre_aug
import getdataset
from opts import opt
from network import get_unet
from uptake_metrics import highest_peak
from read_change_images import normalization, read_image, change_spacing, resize_array, center_crop, closing, remove_small_regions, load_save_dicom
from calc_analysis import calc_analysis
from losses import dice_coef_loss, dice_coef_multilabel
from background import get_background
from pyradiomics import radiomics_feature_extraction
from get_mesh import mesh_analysis


def evaluate():
    """evaluate the UNet model by stepwise moving along the 3D image"""
    print("Please cite our article when you use SEQUOIA: van Praagh GD, Nienhuis PH, Reijrink M, et al. Automated multiclass segmentation, quantification, and visualization of the diseased aorta on hybrid PET/CT–SEQUOIA. Med Phys. 2024; 1-14. https://doi.org/10.1002/mp.16967")
    model = get_unet(img_size=tuple(opt.img_size), num_channels=opt.num_channels, num_classes=opt.num_classes)
    
    # add clipnorm to optimizer to prevent nan values during training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=opt.learning_rate, clipnorm=1), loss=dice_coef_loss, metrics=[dice_coef_multilabel], run_eagerly=True)
    checkpoints = os.listdir(os.path.join(os.path.dirname(__file__), opt.checkpoint_path))
    model_path = os.path.join(os.path.dirname(__file__), opt.checkpoint_path,checkpoints[-1])
    model.load_weights(model_path)
    
    folder_list = [f.path for f in os.scandir(opt.data_dir) if f.is_dir()]
    for path in folder_list:
        cases = []
        column_list = []
        begin_time = datetime.datetime.now()
        print(path)
        output_path = os.path.join(path,'output_SEQUOIA')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        #get images
        if not os.path.exists(os.path.join(path, opt.ct_filename)):
            ct, pet, pet_available = load_save_dicom(path)
            if pet_available:
                pet.SetOrigin(ct.GetOrigin())
                pet.SetDirection(ct.GetDirection())
                pet.SetSpacing(ct.GetSpacing())
            else:
                print("PET filename cannot be found, software will continue without PET calculations")
        else:
            ct = read_image(os.path.join(path, opt.ct_filename))
            if opt.pet_filename == 'none':
                pet_available = False
            else:
                try:
                    pet = read_image(os.path.join(path, opt.pet_filename))
                    pet.SetOrigin(ct.GetOrigin())
                    pet = sitk.Resample(pet, ct)
                    pet_available = True
                except RuntimeError:
                    pet_available = False
                    print("PET filename cannot be found, software will continue without PET calculations")
        
        spacing = ct.GetSpacing()
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        x,y,z = ct.GetSize()
        if opt.cropping:
            if z > 600:
                output_size = (512,512,500)
                index = [0,0,int(ct.GetSize()[2]-600)]
                roiFilter = sitk.RegionOfInterestImageFilter()
                roiFilter.SetSize(output_size)
                roiFilter.SetIndex(index)
                image = roiFilter.Execute(ct)
        image = change_spacing(image, [0.9765625, 0.9765625, 1.5])
        image = center_crop(image)
        image = normalization(image)
        sample = {'image': [image]}
        x_size, y_size, z_size = image.GetSize()

        if opt.speed == 'fast':
            #Less overlap per patch --> faster
            x_list = [0, int(x_size-opt.img_size[0]-1)]
            y_list = [0, int(y_size-opt.img_size[1]-1)]
            z_list = list(range(0,int(z_size-opt.img_size[2]),int(opt.img_size[2]-4)))
        elif opt.speed == 'medium':
            #Fast but more accurate
            x_list = [0, int(x_size-opt.img_size[0]-1)]
            y_list = [0, int(y_size-opt.img_size[1]-1)]
            z_list = list(range(0,int(z_size-opt.img_size[2]),int(opt.img_size[2]/2)))
        elif opt.speed == 'accurate':
            #Most predictions per voxel (more towards the middle)
            x_list = list(range(0,int(x_size-opt.img_size[0]),int(opt.img_size[0]/2)))
            x_list.append(int(x_size-opt.img_size[0]-1))
            y_list = list(range(0,int(y_size-opt.img_size[1]),int(opt.img_size[1]/2)))
            y_list.append(int(y_size-opt.img_size[1]-1))
            z_list = list(range(0,int(z_size-opt.img_size[2]),int(opt.img_size[2]/2)))
            z_list.append(int(z_size-opt.img_size[2]-1))
        
        index_list = [[x, y, z] for z in z_list for y in y_list for x in x_list]
        ind_len = len(index_list)
        if ind_len % opt.batch_size != 0:
            remaining = opt.batch_size - (ind_len - ind_len//opt.batch_size * opt.batch_size)
            if remaining <= opt.batch_size/2 and opt.speed == 'accurate':
                for r in range(remaining + opt.batch_size):
                    index_list.append([int(x_size/2-opt.img_size[0]/2), int(y_size/2-opt.img_size[1]/2), z_list[int((len(z_list)-remaining-opt.batch_size)/2)+r]])
            else:
                for r in range(remaining):
                    index_list.append([int(x_size/2-opt.img_size[0]/2), int(y_size/2-opt.img_size[1]/2), z_list[int((len(z_list)-remaining)/2)+r]])
        index_rounds = int(len(index_list) / opt.batch_size)
        sitk_predictions = []
        full_prediction = np.zeros((z_size,y_size,x_size,opt.num_classes))
        prep_time = datetime.datetime.now()-begin_time
        print("Preprocessing time: {:.2f} sec".format(prep_time.total_seconds()))
        
        pred_time_start = datetime.datetime.now()
        for round in range(index_rounds):
            # create transformations to image
            index = index_list[round*opt.batch_size:opt.batch_size+round*opt.batch_size]
            
            PredictionTransforms = [pre_aug.Crop(opt.img_size, index)]

            EvaluationDataset = getdataset.GetDataset(
                sample=sample,
                input_path=[path],
                batch_size=opt.batch_size,
                img_size=tuple(opt.img_size),
                num_channels=opt.num_channels,
                transforms=PredictionTransforms,
                )
            
            prediction = model.predict(EvaluationDataset)
            for i in range(opt.num_classes):
                pred_right_size = np.zeros((y_size,x_size,z_size,opt.num_classes))
                for b in range(opt.batch_size):
                    prediction_np = prediction[b,:,:,:,:]
                    pred_right_size[index[b][1]:index[b][1]+prediction_np.shape[1], index[b][0]:index[b][0]+prediction_np.shape[0], index[b][2]:index[b][2]+prediction_np.shape[2], 0:opt.num_classes] = prediction_np
                pred_right_size = np.transpose(pred_right_size,(2,0,1,3))
                full_prediction += pred_right_size

        max_indices = np.argmax(full_prediction, axis=-1)
        result = np.zeros_like(full_prediction)
        result[np.arange(result.shape[0])[:, None, None, None], np.arange(result.shape[1])[None, :, None, None], np.arange(result.shape[2])[None, None, :, None], max_indices[:, :, :, None]] = 1
        result = result[:,:,:,1:]
        result = result.astype(np.int8)
        result = closing(result)
        result = resize_array(result, image, ct)

        pred_time = datetime.datetime.now()-pred_time_start
        print("prediction time: {:.2f} sec".format(pred_time.total_seconds()))

        calc_time_start = datetime.datetime.now()
        for channel in range(result.shape[3]):
            pred_np = result[:,:,:,channel]
            min_size = int(10000 / voxel_volume)
            pred_np = remove_small_regions(pred_np, min_size, spacing)
            pred_sitk = sitk.GetImageFromArray(pred_np)
            pred_sitk.SetSpacing(ct.GetSpacing())
            pred_sitk.SetDirection(ct.GetDirection())
            pred_sitk.SetOrigin(ct.GetOrigin())
            pred_sitk = sitk.Resample(pred_sitk, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, ct.GetPixelID())
            sitk_predictions.append(pred_sitk)
            label_names = ['asc_aorta', 'arc_aorta', 'des_aorta', 'abd_aorta']
            sitk.WriteImage(pred_sitk, os.path.join(output_path, '{}.nii.gz'.format(label_names[channel])))
            print("prediction {} saved".format(label_names[channel]))

            if channel == 0:
                asc_background = get_background(pred_sitk)
                sitk.WriteImage(asc_background, os.path.join(output_path, 'asc_background.nii.gz'))
                castFilter = sitk.CastImageFilter()
                castFilter.SetOutputPixelType(sitk.sitkInt8)
                asc_background = castFilter.Execute(asc_background)
                labelFilter = sitk.LabelStatisticsImageFilter()
                labelFilter.Execute(ct, asc_background)
                bg_ct_mean = labelFilter.GetMean(1)
                bg_ct_sd = labelFilter.GetSigma(1)

                if pet_available:
                    labelFilter = sitk.LabelStatisticsImageFilter()
                    labelFilter.Execute(pet, asc_background)
                    bg_pet_mean = labelFilter.GetMean(1)

            pred_volume = np.count_nonzero(pred_np == 1) * voxel_volume

            if pet_available:
                labelFilter = sitk.LabelStatisticsImageFilter()
                labelFilter.Execute(pet, pred_sitk)
                suv_mean_pred, suv_max_pred, suv_sum_pred = labelFilter.GetMean(1), labelFilter.GetMaximum(1), labelFilter.GetSum(1)

                suv_peak_pred = highest_peak(pet, pred_sitk)

            if opt.calcium_threshold == 'standard':
                calcium_threshold = 130
            elif opt.calcium_threshold == '100kVP':
                calcium_threshold = 147
            elif opt.calcium_threshold == 'SD':
                calcium_threshold = bg_ct_mean + 2 * bg_ct_sd
            
            if opt.per_calc:
                if pet_available:
                    ag_pred, vol_pred, mass_pred, calcifications = calc_analysis(ct, pred_sitk, calcium_threshold, per_calc=opt.per_calc, pet=pet, bg_pet_mean=bg_pet_mean)
                    df = pd.DataFrame(calcifications, columns=['nr', 'sl', 'ind', 'area_calc', 'vol_calc', 'AS_calc', 'mass_calc', 'max_HU', 'mean_HU', 'suv_mean_calc', 'suv_max_calc', 'suv_min_calc', 'suv_sum_calc', 'tbr_mean_calc', 'tbr_max_calc'], index=column_list)
                    df.to_csv(os.path.join(output_path,'per_calcification_analysis.csv'))
                else:
                    ag_pred, vol_pred, mass_pred, calcifications = calc_analysis(ct, pred_sitk, calcium_threshold, per_calc=opt.per_calc)
                    df = pd.DataFrame(calcifications, columns=['nr', 'sl', 'ind', 'area_calc', 'vol_calc', 'AS_calc', 'mass_calc', 'max_HU', 'mean_HU'], index=column_list)
                    df.to_csv(os.path.join(output_path,'per_calcification_analysis.csv'))

            else:
                ag_pred, vol_pred, mass_pred = calc_analysis(ct, pred_sitk, calcium_threshold)
            
            if pet_available:
                case = [suv_mean_pred, suv_max_pred, suv_peak_pred, suv_sum_pred, '', pred_volume, '', ag_pred, vol_pred, mass_pred]
            else:
                case = [pred_volume, '', ag_pred, vol_pred, mass_pred]
            
            cases.append(case)
            column_list.append(path + label_names[channel])
            if pet_available:
                df = pd.DataFrame(cases, columns=['suv_mean', 'suv_max', 'suv_peak', 'suv_sum', '', 'volume', '', 'ag', 'vol', 'mass'], index=column_list)
                df.to_csv(os.path.join(output_path,'data.csv'))
            else:
                df = pd.DataFrame(cases, columns=['volume', '', 'ag', 'vol', 'mass'], index=column_list)
                df.to_csv(os.path.join(output_path,'data.csv'))

            if opt.radiomics:
                if pet_available:
                    df2 = radiomics_feature_extraction(pet, pred_sitk)
                    df2.to_csv(os.path.join(output_path,'radiomics_segment{}.csv'.format(label_names[channel])))
                else:
                    df2 = radiomics_feature_extraction(ct, pred_sitk)
                    df2.to_csv(os.path.join(output_path,'radiomics_segment{}.csv'.format(label_names[channel])))
                    print("Radiomics are done on CT image, because PET was not available")

            
        pred_combined = sitk_predictions[0]
        if len(sitk_predictions) > 1:
            for i in range(1,len(sitk_predictions)):
                next_label = sitk_predictions[i] * (i+1)
                pred_combined += next_label
        
        pred_combined.SetSpacing(ct.GetSpacing())
        pred_combined.SetDirection(ct.GetDirection())
        pred_combined.SetOrigin(ct.GetOrigin())

        prediction_original_size = sitk.Resample(pred_combined, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, ct.GetPixelID())
        sitk.WriteImage(prediction_original_size, os.path.join(output_path, 'aorta_combined.nii.gz'))
        pred_combined_np = sitk.GetArrayFromImage(pred_combined)
        pred_combined_np[pred_combined_np>0] = 1
        pred = sitk.GetImageFromArray(pred_combined_np)
        pred.SetSpacing(ct.GetSpacing())
        pred.SetDirection(ct.GetDirection())
        pred.SetOrigin(ct.GetOrigin())
        pred = sitk.Resample(pred, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, ct.GetPixelID())
        sitk.WriteImage(pred, os.path.join(output_path, 'aorta.nii.gz'))

        calc_time = datetime.datetime.now()-calc_time_start
        print("Quantification and saving time: {:.2f} sec".format(calc_time.total_seconds()))
        mesh_time_start = datetime.datetime.now()
        if opt.meshes:
            if pet_available:
                if opt.pet_threshold == 'A50P':
                    highest_mean_peak = highest_peak(pet, pred)
                    pet_threshold = bg_pet_mean + 0.5 * highest_mean_peak
                else:
                    pet_threshold = float(opt.pet_threshold)
                ct_values, pet_values = mesh_analysis(pred, ct, calcium_threshold, output_path, pet, pet_threshold, pet_available)
            else:
                ct_values = mesh_analysis(pred, ct, calcium_threshold, output_path)
            mesh_time = datetime.datetime.now() - mesh_time_start
            print("Mesh time: {:.2f} sec".format(mesh_time.total_seconds()))
        else:
            mesh_time = datetime.datetime.now()-mesh_time_start
        # time = [prep_time, pred_time, calc_time, mesh_time]
        # times.append(time)
        # paths.append(path)
        # df = pd.DataFrame(times, columns=['prep_time', 'pred_time', 'calc_time', 'mesh_time'])
        # df.to_csv(os.path.join(opt.data_dir,'times.csv'))
        print(datetime.datetime.now() - begin_time)
        print("Please cite our article when you use SEQUOIA: van Praagh GD, Nienhuis PH, Reijrink M, et al. Automated multiclass segmentation, quantification, and visualization of the diseased aorta on hybrid PET/CT–SEQUOIA. Med Phys. 2024; 1-14. https://doi.org/10.1002/mp.16967")
    
    return cases


def main(argv=None):
    evaluate()
    
if __name__=='__main__':
    evaluate()
