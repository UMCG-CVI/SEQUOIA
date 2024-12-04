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


import numpy as np
from datetime import datetime


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
    else:
        series_date_time = series_date + series_time
    
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

def convert_to_metric(dcm_dataset, img_volume_array, suv_bw=True, suv_bsa=False, sul=False, patient_height_in_m=None, weight=None):
    try:
        patient_weight_in_kg = dcm_dataset.PatientWeight
    except:
        patient_weight_in_kg = weight
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
            if patient_height_in_m > 100:
                patient_height_in_m = patient_height_in_m / 100
            print("patient_height_in_m", patient_height_in_m)
            
            bsa = patient_weight_in_kg^0.425 * (patient_height_in_m*100)**0.725 * 71.84
            scaled_img_volume_array = img_volume_array * suv_scaling_factor * bsa
        except:
            print("No patient size in dicom header")
            scaled_img_volume_array = 0
    
    if sul:
        try:
            try:
                patient_height_in_m = dcm_dataset.PatientSize
            except:
                patient_height_in_m = patient_height_in_m
            if patient_height_in_m > 100:
                patient_height_in_m = patient_height_in_m / 100
            print("patient_height_in_m", patient_height_in_m)

            BMI = patient_weight_in_kg/(patient_height_in_m**2)
            if patient_gender == 'F':
                LBMjanma = (9270 * patient_weight_in_kg) / (8780 + 244*BMI)
            if patient_gender == 'M':
                LBMjanma = (9270 * patient_weight_in_kg) / (6680 + 216*BMI)
            scaled_img_volume_array = img_volume_array * suv_scaling_factor * LBMjanma * 1000
        except:
            print("No patient size in dicom header")
            scaled_img_volume_array = 0
    
    return scaled_img_volume_array
