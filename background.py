"""
SEQUOIA - automated multiclass SEgmentation, QUantification, and visualizatiOn of the dIseased Aorta on hybrid PET/CT

@author: praaghgd
"""

import SimpleITK as sitk
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def get_background(label):
    # LabelShapeStatisticsImageFilter returns a bounding box as [xstart, ystart, start, xsize, ysize, zsize]
    labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
    labelShapeFilter.Execute(label)

    Bbox = labelShapeFilter.GetBoundingBox(1)
    middle_slice = int(Bbox[2] + Bbox[5]/2)
    label_np = sitk.GetArrayFromImage(label)
    label_np = label_np.astype('uint8')
    middle_slice_np = label_np[middle_slice,:,:]
    middle_slice_np = middle_slice_np.astype(np.uint8)
    contours, _ = cv2.findContours(middle_slice_np,2,1)
    radius2 = 0
    for i in range(len(contours)):
        (x, y), radius = cv2.minEnclosingCircle(contours[i])
        if radius > radius2:
            radius2 = radius
    slice_range = 12/label.GetSpacing()[2]
    kernel_radius = math.floor(radius2*2-12/label.GetSpacing()[0])
    plt.imshow(middle_slice_np)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_radius, kernel_radius))
    background_np = np.zeros(label_np.shape)
    for i in range(math.floor(middle_slice-slice_range/2),math.ceil(middle_slice+slice_range/2)):
        erosion = cv2.erode(label_np[i,:,:], kernel)
        background_np[i,:,:] = erosion
    background = sitk.GetImageFromArray(background_np)
    background.SetSpacing(label.GetSpacing())
    background.SetDirection(label.GetDirection())
    background.SetOrigin(label.GetOrigin())

    # erode = sitk.BinaryErodeImageFilter()
    # erode.SetKernelType(sitk.sitkBall)
    # erode.SetKernelRadius(round(radius2-6/label.GetSpacing()[0])) #zodat er 6 mm overblijft
    # erode.SetForegroundValue(1)
    # background = erode.Execute(label)
    # labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
    # labelShapeFilter.Execute(background)
    # Bbox = labelShapeFilter.GetBoundingBox(1)
    # if Bbox[5] > 6:
    #     background_np = sitk.GetArrayFromImage(background)
    #     new_background_np = np.zeros(background_np.shape)
    #     start_slice = int(Bbox[2] + Bbox[5]/2 - 3)
    #     end_slice = start_slice + 6
    #     new_background_np[start_slice:end_slice,:,:] = background_np[start_slice:end_slice,:,:]
    #     background = sitk.GetImageFromArray(new_background_np)
    #     background.SetSpacing(label.GetSpacing())
    #     background.SetDirection(label.GetDirection())
    #     background.SetOrigin(label.GetOrigin())
    # elif Bbox[5] < 6:
    #     contours, _ = cv2.findContours(middle_slice_np,2,1)
    #     for i in range(len(contours)):
    #         (x, y), radius = cv2.minEnclosingCircle(contours[i])
    #         if i == 0:
    #             radius2 = radius
    #         elif radius < radius2:
    #             radius2 = radius
    #     print(radius2)
    #     erode = sitk.BinaryErodeImageFilter()
    #     erode.SetKernelType(sitk.sitkBall)
    #     erode.SetKernelRadius(round(radius2-6*label.GetSpacing()[0])) #zodat er 6 mm overblijft
    #     erode.SetForegroundValue(1)
    #     background = erode.Execute(label)
    #     labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
    #     labelShapeFilter.Execute(background)
    #     Bbox = labelShapeFilter.GetBoundingBox(1)
    #     if Bbox[5] > 6:
    #         background_np = sitk.GetArrayFromImage(background)
    #         new_background_np = np.zeros(background_np.shape)
    #         start_slice = int(Bbox[2] + Bbox[5]/2 - 3)
    #         end_slice = start_slice + 6
    #         new_background_np[start_slice:end_slice,:,:] = background_np[start_slice:end_slice,:,:]
    #         background = sitk.GetImageFromArray(new_background_np)
    #         background.SetSpacing(label.GetSpacing())
    #         background.SetDirection(label.GetDirection())
    #         background.SetOrigin(label.GetOrigin())
    
    return background
