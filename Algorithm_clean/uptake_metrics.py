"""
SEQUOIA - automated multiclass SEgmentation, QUantification, and visualizatiOn of the dIseased Aorta on hybrid PET/CT


Copyright 2023 University Medical Center Groningen

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
import sys
import pydicom
import numpy as np
from datetime import datetime
import SimpleITK as sitk
from read_change_images import crop_image_to_label, make_isotropic

def read_dicom_files(input_dicom_dir):
    filenames = os.listdir(input_dicom_dir)
    data=True
    while data:
        for filename in filenames:
            try:
                dcm_dataset = pydicom.dcmread(os.path.join(input_dicom_dir, filename))
                data = False
                print("Valid DICOM: " + input_dicom_dir)
                break
            except:
                sys.exit("Not a valid DICOM: " + input_dicom_dir)
    return dcm_dataset


def calculate_decay_constant(dcm_dataset):
    
    half_life_of_radionuclide_in_hr = (dcm_dataset.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)/(60*60) # (0018, 1075)
    print("half_life_of_radionuclide_in_hr", half_life_of_radionuclide_in_hr)
    decay_constant = np.log(2) / half_life_of_radionuclide_in_hr
    print("decay constant", decay_constant)
    return decay_constant

def calculate_decay_factor(dcm_dataset):    
    
    try:
        radiopharmaceutical_start_date_time = dcm_dataset.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartDateTime # 20151103.062600 # injection date time: (0018, 1078) using the same time base as Series Time
    except:
        radiopharmaceutical_start_date_time = dcm_dataset.SeriesDate + dcm_dataset.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
    # Series Date (0008,0021) and Series Time (0008,0031) are used as the reference time for all PET Image Attributes that are temporally related, including activity measurements. 
    # Series Date (0008,0021) and Series Time (0008,0031) are not tied to any real-world event (e.g., acquisition start, radiopharmaceutical administration) and their real-world meaning are implementation dependent.    
    series_date = dcm_dataset.SeriesDate # (0008, 0021)  
    if series_date == '':
        series_date = dcm_dataset.StudyDate
    series_time = dcm_dataset.SeriesTime # (0008, 0031)    
    if series_time == '':
        series_time = dcm_dataset.StudyTime
    if '.' not in series_time:
        series_date_time = series_date + series_time + '.000000' # 20151109.013628 # scan start date time 
    
    # Acquisition Date (0008,0022) and Acquisition Time (0008,0032) use the same time base as Series Time (0008,0031).
    # For Series Type (0054,1000) Value 1 equal to STATIC, WHOLE BODY, or DYNAMIC, Acquisition Time (0008,0032) is the real-world beginning of the accumulation of events into this Image. 
    # For STATIC, WHOLE BODY, or DYNAMIC Series, Acquisition Time (0008,0032) may vary from Image to Image within a PET Series.
    
    # acquisition_date_time = dcm_dataset[0].AcquisitionDateTime # (0008, 002a) (0008, 0022) (0008, 0032)

    fmt = '%Y%m%d%H%M%S.%f' # Change as per date time format in DICOM file # 20151109.013628 or # 20151109013628.000000
    time_stamp_1 = datetime.strptime(radiopharmaceutical_start_date_time, fmt)
    time_stamp_2 = datetime.strptime(series_date_time, fmt)
    print(time_stamp_1, time_stamp_2)
    time_difference = time_stamp_2 - time_stamp_1
    print("time_difference", time_difference)
    if time_stamp_2 < time_stamp_1:
        radiopharmaceutical_start_time = dcm_dataset.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
        series_time = dcm_dataset.SeriesTime # (0008, 0031)    
        if series_time == '':
            series_time = dcm_dataset.StudyTime
        if '.' not in series_time:
            series_time = series_time + '.000000'
        fmt = '%H%M%S.%f'
        time_stamp_1 = datetime.strptime(radiopharmaceutical_start_time, fmt)
        time_stamp_2 = datetime.strptime(series_time, fmt)
        time_difference = time_stamp_2 - time_stamp_1
        if time_stamp_2.time() < time_stamp_1.time():
            print("Something is wrong with the injection time ({}) or series time ({}) in the DICOM header".format(radiopharmaceutical_start_date_time, series_date_time))
        print("Injection time ({}) is later than series time ({}), check DICOM header, only times are used for now". format(radiopharmaceutical_start_date_time, series_date_time))
    time_elapsed_in_hr = (time_difference.total_seconds() / (60*60))
    print("time_elapsed_in_hr", time_elapsed_in_hr)
    
    decay_constant = calculate_decay_constant(dcm_dataset)
    decay_factor = np.exp(-decay_constant * time_elapsed_in_hr)
    return decay_factor

def get_decay_corrected_activity_in_Bq_at_scan_start(dcm_dataset):
    
    injected_activity_in_Bq = float(dcm_dataset.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose) # (0018, 1074)
    print("injected_activity_in_Bq", injected_activity_in_Bq)
    decay_factor = calculate_decay_factor(dcm_dataset)
    decay_corrected_activity_in_Bq_at_scan_start = injected_activity_in_Bq * decay_factor
    return decay_corrected_activity_in_Bq_at_scan_start

def convert_to_metric(dcm_dataset, img_volume_array, suv_bw=True, suv_bsa=False, sul=False):
    patient_weight_in_kg = dcm_dataset.PatientWeight
    print("patient_weight_in_kg", patient_weight_in_kg)
    patient_gender = str(dcm_dataset.PatientSex)
    print("patient_gender", patient_gender)
    decay_corrected_activity_in_Bq_at_scan_start = get_decay_corrected_activity_in_Bq_at_scan_start(dcm_dataset)
    print("decay_corrected_activity_in_Bq_at_scan_start", decay_corrected_activity_in_Bq_at_scan_start)
    suv_scaling_factor = 1/(decay_corrected_activity_in_Bq_at_scan_start)
    print("suv_scaling_factor", suv_scaling_factor)
    
    if suv_bw:
        scaled_img_volume_array = img_volume_array * suv_scaling_factor * patient_weight_in_kg * 1000
    
    if suv_bsa:
        try:
            patient_height_in_m = dcm_dataset.PatientSize
            bsa = patient_weight_in_kg^0.425 * (patient_height_in_m*100)^0.725 * 71.84
            scaled_img_volume_array = img_volume_array * suv_scaling_factor * bsa
        except:
            print("No patient size in dicom header")
            scaled_img_volume_array = 0
    
    if sul:
        try:
            patient_height_in_m = dcm_dataset.PatientSize
            BMI = patient_weight_in_kg/(patient_height_in_m^2)
            if patient_gender == 'F':
                LBMjanma = (9270 * patient_weight_in_kg) / (8780 + 244*BMI)
            if patient_gender == 'M':
                LBMjanma = (9270 * patient_weight_in_kg) / (6680 + 216*BMI)
            scaled_img_volume_array = img_volume_array * suv_scaling_factor * LBMjanma * 1000
        except:
            print("No patient size in dicom header")
            scaled_img_volume_array = 0
    
    return scaled_img_volume_array

def calculate_uptake_metrics(scaled_img_volume_array):

    try:
        suv_scaled_img_np = sitk.GetArrayFromImage(scaled_img_volume_array) # comment this sentence if you don't use sitk, but numpy
    except:
        pass
    try:
        suv_mean = suv_scaled_img_np[suv_scaled_img_np!=0].mean()
    except:
        suv_mean = 0
    try:
        suv_max = suv_scaled_img_np[suv_scaled_img_np!=0].max()
    except:
        suv_max = 0
    try:
        suv_min = suv_scaled_img_np[suv_scaled_img_np!=0].min()
    except:
        suv_min = 0
    try:
        suv_sum = suv_scaled_img_np[suv_scaled_img_np!=0].sum()
    except:
        suv_sum = 0
    
    return suv_mean, suv_max, suv_min, suv_sum


def highest_peak(pet, label):
    def highest_peak_mask(pixelspacing):
        def downsample(a):
            q = np.array([a[::2,::2,::2], a[1::2,::2,::2],
                        a[::2,1::2,::2], a[::2,::2,1::2],
                        a[1::2,1::2,::2], a[1::2,::2,1::2],
                        a[::2,1::2,1::2], a[1::2,1::2,1::2]])
            return np.mean(q, axis=0)
        
        #pixelspacing should be isotropic
        radius_aa = int((6/pixelspacing)*2) #radius of sphere is 6 mm. For anti-aliasing, multiply everything with 2
        if radius_aa % 2 == 0:
            height_aa, width_aa, depth_aa = (radius_aa+1)*2, (radius_aa+1)*2, (radius_aa+1)*2
        else:
            height_aa, width_aa, depth_aa = radius_aa*2, radius_aa*2, radius_aa*2
        y_aa, x_aa, z_aa = np.indices((height_aa, width_aa, depth_aa))
        mask = ((x_aa+0.5)-width_aa/2)**2 + ((y_aa+0.5)-height_aa/2)**2 + ((z_aa+0.5)-depth_aa/2)**2 < radius_aa**2 #makes boolean of everything within the radius in 3D
        mask = downsample(mask)
        # plt.imshow(mask[...,10])
        
        return mask
    
    cropped_pet, cropped_label = crop_image_to_label(pet, label)
    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(pet.GetPixelID())
    cropped_label = castFilter.Execute(cropped_label)
    mult_mask_pet = cropped_pet * cropped_label
    pet_isotropic = make_isotropic(mult_mask_pet)
    pixelspacing = pet_isotropic.GetSpacing()[0]
    pet_np = sitk.GetArrayFromImage(pet_isotropic)
    pet_np = np.transpose(pet_np, (1,2,0))
    
    mask = highest_peak_mask(pixelspacing)
    highest_mean_peak = 0
    for p in range(pet_np.shape[0]-mask.shape[0]):
        for q in range(pet_np.shape[1]-mask.shape[1]):
            for r in range(pet_np.shape[2]-mask.shape[2]):
                pet_crop = pet_np[p:p+mask.shape[0], q:q+mask.shape[1], r:r+mask.shape[2]]
                pet_mask = pet_crop * mask
                mean_sphere = pet_mask[pet_mask!=0].mean()
                if mean_sphere > highest_mean_peak:
                    highest_mean_peak = mean_sphere
                    
    return highest_mean_peak

