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

import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import datetime
from opts import opt
from uptake_metrics import highest_peak
from read_change_images import read_image, load_save_dicom
from calc_analysis import calc_analysis
from background import get_background
from pyradiomics import radiomics_feature_extraction
from get_mesh import mesh_analysis


def evaluate():
    """evaluate the UNet model by stepwise moving along the 3D image"""
    print("Please cite our article when you use SEQUOIA: van Praagh GD, Nienhuis PH, Reijrink M, et al. Automated multiclass segmentation, quantification, and visualization of the diseased aorta on hybrid PET/CT–SEQUOIA. Med Phys. 2024; 1-14. https://doi.org/10.1002/mp.16967")
    
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
            try:
                ct, pet, pet_available = load_save_dicom(path)
            except UnboundLocalError:
                print("No CT is found! A CT is needed to run SEQUOIA. Please add the CT file(s) to the folder")
                continue
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
        # x,y,z = ct.GetSize()
        # if opt.cropping:
        #     if z > 600:
        #         output_size = (512,512,500)
        #         index = [0,0,int(ct.GetSize()[2]-600)]
        #         roiFilter = sitk.RegionOfInterestImageFilter()
        #         roiFilter.SetSize(output_size)
        #         roiFilter.SetIndex(index)
        #         ct = roiFilter.Execute(ct)
        
        sitk_predictions = []
        label_names = ['asc_aorta', 'arc_aorta', 'des_aorta', 'abd_aorta']
        calc_time_start = datetime.datetime.now()
        for channel in range(len(label_names)):
            pred_sitk = sitk.ReadImage(os.path.join(output_path, '{}.nii.gz'.format(label_names[channel])))
            pred_np = sitk.GetArrayFromImage(pred_sitk)
            sitk_predictions.append(pred_sitk)

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
        print(datetime.datetime.now() - begin_time)
        print("Please cite our article when you use SEQUOIA: van Praagh GD, Nienhuis PH, Reijrink M, et al. Automated multiclass segmentation, quantification, and visualization of the diseased aorta on hybrid PET/CT–SEQUOIA. Med Phys. 2024; 1-14. https://doi.org/10.1002/mp.16967")
    
    return cases


def main(argv=None):
    evaluate()
    
if __name__=='__main__':
    evaluate()
