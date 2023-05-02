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

import SimpleITK as sitk
import numpy as np
import math
import cv2
from read_change_images import crop_image_to_label, binary_threshold

# #DIT MOET PER CALCIFICATIE KUNNEN!
# def calc_analysis(ct, label, calcium_threshold):
#     def AS_weight_calc(mask, calcium_image_slice):
#         mask_ct = mask * calcium_image_slice
#         max_HU = mask_ct[mask_ct!=0].max()
#         mean_HU = mask_ct[mask_ct!=0].mean()
        
#         if max_HU >= 400:
#             AS_weight = 4
#         elif max_HU >= 300:
#             AS_weight = 3
#         elif max_HU >= 200:
#             AS_weight = 2
#         else:
#             AS_weight = 1
    
#         return AS_weight, max_HU, mean_HU
    
#     cropped_ct, cropped_label = crop_image_to_label(ct, label)
#     castFilter = sitk.CastImageFilter()
#     castFilter.SetOutputPixelType(cropped_ct.GetPixelID())
#     cropped_label = castFilter.Execute(cropped_label)
#     mult_mask_ct = cropped_ct * cropped_label
#     thres_ct = binary_threshold(mult_mask_ct, calcium_threshold)

#     thres_ct_np = sitk.GetArrayFromImage(thres_ct)
#     thres_ct_np = np.transpose(thres_ct_np,(1,2,0))
#     cropped_label_np = sitk.GetArrayFromImage(cropped_label)
#     cropped_label_np = np.transpose(cropped_label_np,(1,2,0))
#     cropped_label_np = cropped_label_np.astype(dtype=np.uint8)
#     ct_np = sitk.GetArrayFromImage(cropped_ct)
#     ct_np = np.transpose(ct_np,(1,2,0))
    
#     spacing_x, spacing_y, spacing_z = ct.GetSpacing()

#     #CHANGE PER MANUFACTURER
#     min_area_in_pixels = math.ceil(1 / (spacing_x * spacing_y)) # calcification should be larger than 1 mm2
    
#     total_agatston = 0
#     total_volume = 0
#     total_mass = 0
#     cal_factor = 0.00079 # standard calibration factor of Siemens - Syngo.Via (gives mass in mg)
    
#     indices = np.indices(thres_ct_np.shape[0:2])
#     indices = np.transpose(indices,(1,2,0))
#     for sl in range(thres_ct_np.shape[2]): #do connected component analysis per slice and find calcifications that are larger than minimum required area
#         retval, labs, stats, centroids = cv2.connectedComponentsWithStats(thres_ct_np[...,sl], connectivity=8, ltype=cv2.CV_32S)
#         calc_values = [row for row in range(len(stats)) if stats[row,4] >= min_area_in_pixels]
#         calc_values = calc_values[1:]
        
#         if len(calc_values) > 0:
#             area_calc = (stats[calc_values,4] * spacing_x * spacing_y).sum()
#             vol_calc = area_calc * spacing_z # volume = number of pixels * pixelsize^2 * slice thickness
#             labs[labs>0] = 1
#             AS_weight, max_HU, mean_HU = AS_weight_calc(labs, ct_np[...,sl]) # get Agatston weightfactor of that calcification
#             AS_calc = (AS_weight * area_calc) / (3 / spacing_z) # Agatston score = weightfactor * area of calcification / correction for slice thickness (as regular Agatston score are calculated with 3 mm slices)
                
#             mass_calc = cal_factor * vol_calc * mean_HU
            
#             total_mass += mass_calc
#             total_agatston += AS_calc
#             total_volume += vol_calc
            
#     return total_agatston, total_volume, total_mass


# def calc_analysis(ct, label, calcium_threshold):
#     def AS_weight_calc(mask, calcium_image_slice):
#         mask_ct = mask * calcium_image_slice
#         max_HU = mask_ct[mask_ct!=0].max()
#         mean_HU = mask_ct[mask_ct!=0].mean()
        
#         if max_HU >= 400:
#             AS_weight = 4
#         elif max_HU >= 300:
#             AS_weight = 3
#         elif max_HU >= 200:
#             AS_weight = 2
#         else:
#             AS_weight = 1
    
#         return AS_weight, max_HU, mean_HU
    
#     cropped_ct, cropped_label = crop_image_to_label(ct, label)
#     castFilter = sitk.CastImageFilter()
#     castFilter.SetOutputPixelType(cropped_ct.GetPixelID())
#     cropped_label = castFilter.Execute(cropped_label)
#     mult_mask_ct = cropped_ct * cropped_label
#     thres_ct = binary_threshold(mult_mask_ct, calcium_threshold)

#     thres_ct_np = sitk.GetArrayFromImage(thres_ct)
#     thres_ct_np = np.transpose(thres_ct_np,(2,0,1)) # swap the axes to have the slices as the first dimension
#     cropped_label_np = sitk.GetArrayFromImage(cropped_label)
#     cropped_label_np = np.transpose(cropped_label_np,(2,0,1)) # swap the axes to have the slices as the first dimension
#     cropped_label_np = cropped_label_np.astype(dtype=np.uint8)
#     ct_np = sitk.GetArrayFromImage(cropped_ct)
#     ct_np = np.transpose(ct_np,(2,0,1)) # swap the axes to have the slices as the first dimension
    
#     spacing_x, spacing_y, spacing_z = ct.GetSpacing()

#     cal_factor = 0.00079 # standard calibration factor of Siemens - Syngo.Via (gives mass in mg)

#     #CHANGE PER MANUFACTURER
#     min_area_in_pixels = math.ceil(1 / (spacing_x * spacing_y)) # calcification should be larger than 1 mm2
    
#     # Apply connectedComponentsWithStats to each slice of the 3D array
#     cc_results = np.apply_along_axis(lambda x: cv2.connectedComponentsWithStats(x, connectivity=8, ltype=cv2.CV_32S), axis=0, arr=thres_ct_np)
    
#     calc_values = np.where(cc_results[3][:,1:,4] >= min_area_in_pixels) # find calcifications that are larger than minimum required area
#     areas_calc = cc_results[3][calc_values[0]+1,calc_values[1]+1,4] * spacing_x * spacing_y
#     vol_calc = areas_calc * spacing_z # volume = number of pixels * pixelsize^2 * slice thickness
#     cc_labels = np.zeros_like(thres_ct_np, dtype=np.uint8)
#     cc_labels[cc_results[1] > 0] = 1
#     AS_weight, max_HU, mean_HU = AS_weight_calc(cc_labels, ct_np) # get Agatston weightfactor of that calcification
#     AS_calc = (AS_weight * areas_calc.sum()) / (3 / spacing_z) # Agatston score = weightfactor * area of calcification / correction for slice thickness (as regular Agatston score are calculated with 3 mm slices)
#     mass_calc = cal_factor * vol_calc.sum() * mean_HU
    
#     return AS_calc, vol_calc, mass_calc


def calc_analysis(ct, label, calcium_threshold, per_calc=False, pet=None, bg_pet_mean=None):
    def AS_weight_calc(mask, calcium_image_slice):
        mask_ct = mask * calcium_image_slice
        max_HU = mask_ct[mask_ct!=0].max()
        mean_HU = mask_ct[mask_ct!=0].mean()
        
        if max_HU >= 400:
            AS_weight = 4
        elif max_HU >= 300:
            AS_weight = 3
        elif max_HU >= 200:
            AS_weight = 2
        else:
            AS_weight = 1
    
        return AS_weight, max_HU, mean_HU
    
    if pet:
        cropped_pet, cropped_label = crop_image_to_label(pet, label)
    cropped_ct, label = crop_image_to_label(ct, label)
    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(cropped_ct.GetPixelID())
    label = castFilter.Execute(label)
    mult_mask_ct = cropped_ct * label
    thres_ct = binary_threshold(mult_mask_ct, calcium_threshold)

    thres_np = sitk.GetArrayFromImage(thres_ct)
    thres_np = np.transpose(thres_np,(1,2,0))
    thres_np = thres_np.astype(dtype=np.uint8)
    ct_np = sitk.GetArrayFromImage(cropped_ct)
    ct_np = np.transpose(ct_np,(1,2,0))
    if pet:
        pet_np = sitk.GetArrayFromImage(cropped_pet)
        pet_np = np.transpose(pet_np,(1,2,0))
    
    spacing_x, spacing_y, spacing_z = ct.GetSpacing()
    min_area_in_pixels = math.ceil(1 / (spacing_x * spacing_y)) # calcification should be larger than 1 mm2
    
    calcifications = []
    total_agatston = 0
    total_volume = 0
    total_mass = 0
    nr = 0
    cal_factor = 0.00079 # standard calibration factor of Siemens - Syngo.Via (gives mass in mg)
    
    indices = np.indices(thres_np.shape[0:2])
    indices = np.transpose(indices,(1,2,0))
    for sl in range(thres_np.shape[2]): #do connected component analysis per slice and find calcifications that are larger than minimum required area
        retval, labs, stats, centroids = cv2.connectedComponentsWithStats(thres_np[...,sl], connectivity=8, ltype=cv2.CV_32S)
        calc_values = [row for row in range(len(stats)) if stats[row,4] >= min_area_in_pixels]
        calc_values = calc_values[1:]
        for calc in calc_values: #do analysis for all calcifications
            area_calc = stats[calc,4] * spacing_x * spacing_y
            vol_calc = area_calc * spacing_z # volume = number of pixels * pixelsize^2 * slice thickness
            total_volume += vol_calc
            
            mask = np.zeros(labs.shape, dtype=np.uint8) 
            mask[labs==calc] = 1 # get mask of only the specific calcification
            ind = indices[mask==1]
            AS_weight, max_HU, mean_HU = AS_weight_calc(mask, ct_np[...,sl]) # get Agatston weightfactor of that calcification
            AS_calc = (AS_weight * area_calc) / (3 / spacing_z) # Agatston score = weightfactor * area of calcification / correction for slice thickness (as regular Agatston score are calculated with 3 mm slices)
            total_agatston += AS_calc
            
            mass_calc = cal_factor * vol_calc * mean_HU
            total_mass += mass_calc
            if per_calc:
                if pet:
                    pet_mask = mask * pet_np[...,sl]
                    suv_mean_calc = pet_mask[pet_mask!=0].mean()
                    tbr_mean_calc = suv_mean_calc / bg_pet_mean 
                    suv_max_calc = pet_mask[pet_mask!=0].max()
                    tbr_max_calc = suv_max_calc / bg_pet_mean
                    suv_min_calc = pet_mask[pet_mask!=0].min()
                    suv_sum_calc = pet_mask[pet_mask!=0].sum()
                    calcification = [nr, sl, ind, area_calc, vol_calc, AS_calc, mass_calc, max_HU, mean_HU, suv_mean_calc, suv_max_calc, suv_min_calc, suv_sum_calc, tbr_mean_calc, tbr_max_calc]
                else:
                    calcification = [nr, sl, ind, area_calc, vol_calc, AS_calc, mass_calc, max_HU, mean_HU]
                calcifications.append(calcification)
    
    if per_calc:
        # First slice has no calcifications yet to compare overlap
        for calc_nr in range(len(calcifications)):
            if calcifications[calc_nr][1] == 0:
                calcifications[calc_nr][0] = nr
                nr += 1
            else:
                calc_last_slice = [calcifications[q] for q in range(len(calcifications)) if calcifications[q][1] == calcifications[calc_nr][1]-1]
                overlap = False
                if len(calc_last_slice) > 0:
                    ind = calcifications[calc_nr][2]
                    t = 0
                    for q in range(len(calc_last_slice)):
                        for r in range(len(ind)):
                            for s in range(len(calc_last_slice[q][2])):
                                if all(ind[r,:] == calc_last_slice[q][2][s,:]):
                                    t += 1
                                    if t == 1:
                                        calcifications[calc_nr][0] = calc_last_slice[q][0]
                                    elif t > 1:
                                        for c in range(calc_nr):
                                            if calcifications[c][0] == calc_last_slice[q][0]:
                                                calcifications[c][0] = calcifications[calc_nr][0]
                                    overlap = True
                if overlap == False:
                    calcifications[calc_nr][0] = nr
                    nr += 1
        
        
        calcifications_final = []
        for number in range(nr):
            calc_last_slice = [calcifications[q] for q in range(len(calcifications)) if calcifications[q][0] == number]
            if len(calc_last_slice) == 1:
                calcifications_final.append(calc_last_slice[0])
            elif len(calc_last_slice) > 1:
                for s in range(1,len(calc_last_slice)):
                    if calc_last_slice[s][3] > calc_last_slice[0][3]:
                        calc_last_slice[0][3] = calc_last_slice[s][3] # maximum area is taken
                    calc_last_slice[0][4] += calc_last_slice[s][4] # volume is added
                    calc_last_slice[0][5] += calc_last_slice[s][5] # Agatston score is added
                    calc_last_slice[0][6] += calc_last_slice[s][6] # Mass score is added
                    if calc_last_slice[s][7] > calc_last_slice[0][7]:
                        calc_last_slice[0][7] = calc_last_slice[s][7] # maximum HU is taken
                    calc_last_slice[0][8] = (len(calc_last_slice[0][2]) * calc_last_slice[0][8] + len(calc_last_slice[s][2]) * calc_last_slice[s][8]) / (len(calc_last_slice[0][2]) + len(calc_last_slice[s][2])) # mean HU is calculated by multiplying mean HU with nr of pixels for both slices divided by total nr of pixels
                    if pet:
                        calc_last_slice[0][9] = (len(calc_last_slice[0][2]) * calc_last_slice[0][9] + len(calc_last_slice[s][2]) * calc_last_slice[s][9]) / (len(calc_last_slice[0][2]) + len(calc_last_slice[s][2])) # mean SUV is calculated by multiplying mean SUV with nr of pixels for both slices divided by total nr of pixels
                        if calc_last_slice[s][10] > calc_last_slice[0][10]:
                            calc_last_slice[0][10] = calc_last_slice[s][10] # max SUV is taken
                        if calc_last_slice[s][11] < calc_last_slice[0][11]:
                            calc_last_slice[0][11] = calc_last_slice[s][11] # min SUV is taken
                        calc_last_slice[0][12] += calc_last_slice[s][12] # sum SUV is added
                        if s == len(calc_last_slice)-1:
                            calc_last_slice[0][13] = calc_last_slice[0][10] / bg_pet_mean #TBR calculated again
                calcifications_final.append(calc_last_slice[0])
        
        return total_agatston, total_volume, total_mass, calcifications_final
    
    else:
        return total_agatston, total_volume, total_mass