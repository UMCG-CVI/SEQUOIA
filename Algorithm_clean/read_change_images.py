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
from opts import opt
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy import ndimage
from skimage.transform import resize
from hausdorff import compute_surface_distances



def read_image(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    
    return reader.Execute()

def normalization(image, windowCT=(-200,400)):
    cast = sitk.CastImageFilter()
    cast.SetOutputPixelType(sitk.sitkFloat64)
    
    normalizing = sitk.IntensityWindowingImageFilter()
    normalizing.SetOutputMaximum(1)
    normalizing.SetOutputMinimum(0)
    
    # CT normalization
    normalizing.SetWindowMaximum(windowCT[1])
    normalizing.SetWindowMinimum(windowCT[0])
    image = cast.Execute(image)
    image = normalizing.Execute(image)

    return image


def binary_threshold(image, threshold):
    thresfilter = sitk.BinaryThresholdImageFilter()
    thresfilter.SetLowerThreshold(threshold)
    thresfilter.SetUpperThreshold(3000)
    thresfilter.SetOutsideValue(0)
    thresfilter.SetInsideValue(1)
    return thresfilter.Execute(image)

def image_threshold(image, threshold):
    thresfilter = sitk.ThresholdImageFilter()
    thresfilter.SetLower(threshold)
    thresfilter.SetUpper(3000)
    thresfilter.SetOutsideValue(0)
    return thresfilter.Execute(image)


def crop_image_to_label(main_image, label):
    # guarantee label type to be integer
    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(sitk.sitkInt16)
    label = castFilter.Execute(label)
    # LabelShapeStatisticsImageFilter returns a bounding box as [xstart, ystart, start, xsize, ysize, zsize]
    labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
    labelShapeFilter.Execute(label)
    # LabelStatisticsImageFilter returns a bounding box as [xmin, xmax, ymin, ymax, zmin, zmax]
    labelFilter = sitk.LabelStatisticsImageFilter()
    labelFilter.Execute(main_image, label)

    if labelFilter.GetNumberOfLabels() == 1:
        print("Error: There are no labels")
    elif labelFilter.GetNumberOfLabels() == 2:
        selectedBbox = labelShapeFilter.GetBoundingBox(1)
    else: #if there are more labels, first find the min and max of x,y,z and then create 1 boundingbox of all labels together
        Bbox = list(labelFilter.GetBoundingBox(1))
        for i in range(2,labelFilter.GetNumberOfLabels()):
            Bbox2 = list(labelFilter.GetBoundingBox(i))
            for nr in range(len(Bbox)):
                if nr % 2 == 0 and Bbox2[nr] < Bbox[nr]:
                    Bbox[nr] = Bbox2[nr]
                elif nr % 2 != 0 and Bbox2[nr] > Bbox[nr]:
                    Bbox[nr] = Bbox2[nr]
        selectedBbox = [Bbox[0], Bbox[2], Bbox[4], Bbox[1]-Bbox[0]+1, Bbox[3]-Bbox[2]+1, Bbox[5]-Bbox[4]+1]
            
    roiFilter = sitk.RegionOfInterestImageFilter()
    roiFilter.SetSize(selectedBbox[3:])
    roiFilter.SetIndex(selectedBbox[:3])
    
    cropped_main = roiFilter.Execute(main_image)
    cropped_main.SetSpacing(main_image.GetSpacing())
    cropped_main.SetDirection(main_image.GetDirection())
    cropped_main.SetOrigin(main_image.GetOrigin())
    
    cropped_label = roiFilter.Execute(label)
    cropped_label.SetSpacing(main_image.GetSpacing())
    cropped_label.SetDirection(main_image.GetDirection())
    cropped_label.SetOrigin(main_image.GetOrigin())
    
    return cropped_main, cropped_label


def center_crop(image):
    '''
    Crops the images and labels towards the center to remove unnecessary data.
    First checks if x/y larger than 'standard' 512 and resamples to 512 if needed.
    If images smaller than patch size, it resamples the images to 256.

    Input and output: dictionary of image and label, and sitk pet image
    '''

    spacing = image.GetSpacing()
    x,y,z = image.GetSize()
    if z > 700:
        #if the image contains more than 700 slices, the legs up to slice 700 will be removed
        z = 700
    output_size = (256,256,z)

    if x > 512 or y > 512:
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputSpacing(spacing)
        resampler.SetSize((512,512,image.GetSize()[2]))

        # resample on image
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        # print("Resampling image...")
        image = resampler.Execute(image)
        
    if x > output_size[0] and y > output_size[1]:
        index = [int((x/2-output_size[0]/2)), int((y/2-output_size[1]/2)),int(image.GetSize()[2]-z)]
        roiFilter = sitk.RegionOfInterestImageFilter()
        roiFilter.SetSize(output_size)
        roiFilter.SetIndex(index)
        image = roiFilter.Execute(image)

    elif x < opt.img_size[0] or y < opt.img_size[1]:
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputSpacing(spacing)
        resampler.SetSize(output_size)

        # resample on image
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        # print("Resampling image...")
        image = resampler.Execute(image)

    return image


def change_spacing(image, new_spacing):
    orig_spacing = list(image.GetSpacing())
    if abs(orig_spacing[0]/new_spacing[0] - 1) < 0.1 or abs(orig_spacing[1]/new_spacing[1] - 1) < 0.1 or abs(orig_spacing[2]/new_spacing[2] - 1) < 0.1:
        print('same spacing')
        return image
    else:
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetOutputSpacing(new_spacing)

        orig_size = np.array(image.GetSize(), dtype=int)
        new_size = [int(np.ceil(orig_size[dim] * (orig_spacing[dim]/new_spacing[dim]))) for dim in range(len(orig_size))]
        resample.SetSize(new_size)
        print('Resampled')
        return resample.Execute(image)

def change_spacing2(image, new_spacing, aorta):
    orig_spacing = list(image.GetSpacing())
    if abs(orig_spacing[0]/new_spacing[0] - 1) < 0.1 or abs(orig_spacing[1]/new_spacing[1] - 1) < 0.1 or abs(orig_spacing[2]/new_spacing[2] - 1) < 0.1:
        print('same spacing')
        return image
    else:
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetOutputSpacing(new_spacing)

        orig_size = np.array(image.GetSize(), dtype=int)
        new_size = [int(np.ceil(orig_size[dim] * (orig_spacing[dim]/new_spacing[dim]))) for dim in range(len(orig_size))]
        resample.SetSize(new_size)
        print('Resampled')
        return resample.Execute(image)
    
def resize_array(labels, image, ct):
    aorta = np.zeros_like(labels[:,:,:,0])
    ct_np = sitk.GetArrayFromImage(ct)
    labels2 = np.zeros((ct_np.shape[0], ct_np.shape[1], ct_np.shape[2], labels.shape[3]))
    for channel in range(labels.shape[3]):
        pred_np = labels[:,:,:,channel]
        aorta += pred_np
        pred_sitk = sitk.GetImageFromArray(pred_np)
        pred_sitk.SetSpacing(image.GetSpacing())
        pred_sitk.SetDirection(image.GetDirection())
        pred_sitk.SetOrigin(image.GetOrigin())
        pred_sitk = sitk.Resample(pred_sitk, image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, image.GetPixelID())
        pred_sitk = change_spacing(pred_sitk, ct.GetSpacing())
        pred_sitk = sitk.Resample(pred_sitk, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, ct.GetPixelID())
        pred_np = sitk.GetArrayFromImage(pred_sitk)
        labels2[:,:,:,channel] = pred_np
        if channel == labels.shape[3]-1:
            pred_sitk = sitk.GetImageFromArray(aorta)
            pred_sitk.SetSpacing(image.GetSpacing())
            pred_sitk.SetDirection(image.GetDirection())
            pred_sitk.SetOrigin(image.GetOrigin())
            pred_sitk = sitk.Resample(pred_sitk, image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, image.GetPixelID())
            pred_sitk = change_spacing(pred_sitk, ct.GetSpacing())
            pred_sitk = sitk.Resample(pred_sitk, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, ct.GetPixelID())
            aorta_ol = sitk.GetArrayFromImage(pred_sitk)
            aorta_ol = aorta_ol.astype(np.uint8)
            aorta_ml = labels2[:,:,:,0] + labels2[:,:,:,1] + labels2[:,:,:,2] + labels2[:,:,:,3]
            aorta_ml = aorta_ml.astype(np.uint8)
            added_voxel_indices = np.transpose(np.nonzero(aorta_ol & ~aorta_ml))
            print(len(added_voxel_indices))
            # Initialize an empty numpy array to store the distances
            distances = np.zeros((len(added_voxel_indices), labels.shape[3]))

            # Loop over the labels in the aorta numpy array
            for i in range(labels.shape[3]):
                # Find the indices of the non-zero voxels in the current label
                try:
                    label_indices = np.transpose(np.nonzero(labels[:, :, :, i]))
                except:
                    label_indices = np.transpose(np.nonzero(labels[:, :, ::2, i]))
                # Calculate the distances between the added voxels and the non-zero voxels in the current label
                label_distances = cdist(added_voxel_indices, label_indices, metric='euclidean')
                # Take the minimum distance for each added voxel and store it in the distances array
                distances[:, i] = np.min(label_distances, axis=1)

            # Find the index of the closest label for each added voxel using argmin
            closest_label_indices = np.argmin(distances, axis=1)

            # Assign each added voxel to the closest label in the aorta numpy array
            for i in range(len(added_voxel_indices)):
                closest_label = closest_label_indices[i]
                labels2[added_voxel_indices[i][0], added_voxel_indices[i][1], added_voxel_indices[i][2], closest_label] = 1
            
    return labels2

def closing(labels):
    aorta = labels[:,:,:,0] + labels[:,:,:,1] + labels[:,:,:,2] + labels[:,:,:,3]
    aorta = aorta.astype(np.uint8)
    # Perform closing morphology to fill holes in the aorta numpy array
    # closed_aorta = ndimage.binary_closing(aorta, structure=np.ones((10, 10, 10))).astype(int)
    closed_aorta = cv2.morphologyEx(aorta, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)))

    # Find the indices of the added voxels in the closed_aorta numpy array
    added_voxel_indices = np.transpose(np.nonzero(closed_aorta & ~aorta))

    # Initialize an empty numpy array to store the distances
    distances = np.zeros((len(added_voxel_indices), labels.shape[3]))

    # Loop over the labels in the aorta numpy array
    for i in range(labels.shape[3]):
        # Find the indices of the non-zero voxels in the current label
        try:
            label_indices = np.transpose(np.nonzero(labels[:, :, :, i]))
        except:
            label_indices = np.transpose(np.nonzero(labels[:, :, ::2, i]))
        # Calculate the distances between the added voxels and the non-zero voxels in the current label
        label_distances = cdist(added_voxel_indices, label_indices, metric='euclidean')
        # Take the minimum distance for each added voxel and store it in the distances array
        distances[:, i] = np.min(label_distances, axis=1)

    # Find the index of the closest label for each added voxel using argmin
    closest_label_indices = np.argmin(distances, axis=1)

    # Assign each added voxel to the closest label in the aorta numpy array
    for i in range(len(added_voxel_indices)):
        closest_label = closest_label_indices[i]
        labels[added_voxel_indices[i][0], added_voxel_indices[i][1], added_voxel_indices[i][2], closest_label] = 1

    return labels

def remove_small_regions(image, min_size, spacing):
    labels, num_features = ndimage.label(image)
    if num_features > 1:
        for i in range(1, num_features+1):
            region_size = np.count_nonzero(labels == i)
            if region_size < min_size:
                labels[labels == i] = 0
    labels[labels > 0] = 1

    lab_array, num_features = ndimage.label(labels)
    if num_features > 1:
        for f in range(1, num_features):
            mask1 = np.copy(lab_array)
            mask2 = np.copy(lab_array)
            mask1[mask1!=f] = 0
            mask1[mask1>0] = 1
            mask2[mask2!=f+1] = 0
            mask2[mask2>0] = 1
            surface_distances = compute_surface_distances(mask1, mask2, spacing)
            min_surface_distance = min(min(surface_distances["distances_gt_to_pred"]),min(surface_distances["distances_pred_to_gt"]))
            if min_surface_distance > 100:
                volume1 = np.count_nonzero(lab_array == f)
                volume2 = np.count_nonzero(lab_array == f+1)
                if volume2 < volume1:
                    lab_array[lab_array == f+1] = 0
                elif volume1 < volume2:
                    lab_array[lab_array == f] = 0
                lab_array[lab_array > 0] = 1
                labels = lab_array
    lab_array, num_features = ndimage.label(labels)
    if num_features > 1:
        print("Segments has multiple labels, please check the segment output before interpreting the results!")
    return labels

def make_isotropic(image, interpolator=sitk.sitkLinear):
    '''
    Resample an image to isotropic pixels (using smallest spacing from original) and save to file. Many file formats 
    (jpg, png,...) expect the pixels to be isotropic. By default the function uses a linear interpolator. For
    label images one should use the sitkNearestNeighbor interpolator so as not to introduce non-existant labels.
    '''
    original_spacing = image.GetSpacing()
    # Image is already isotropic, just return a copy.
    if all(spc == original_spacing[0] for spc in original_spacing):
        return sitk.Image(image)
    # Make image isotropic via resampling.
    original_size = image.GetSize()
    min_spacing = min(original_spacing)
    new_spacing = [min_spacing]*image.GetDimension()
    new_size = [int(round(osz*ospc/min_spacing)) for osz,ospc in zip(original_size, original_spacing)]
    return sitk.Resample(image, new_size, sitk.Transform(), interpolator,
                         image.GetOrigin(), new_spacing, image.GetDirection(), 0,
                         image.GetPixelID())