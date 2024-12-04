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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_radius, kernel_radius))
    background_np = np.zeros(label_np.shape)
    for i in range(math.floor(middle_slice-slice_range/2),math.ceil(middle_slice+slice_range/2)):
        erosion = cv2.erode(label_np[i,:,:], kernel)
        background_np[i,:,:] = erosion
    background = sitk.GetImageFromArray(background_np)
    background.SetSpacing(label.GetSpacing())
    background.SetDirection(label.GetDirection())
    background.SetOrigin(label.GetOrigin())

    return background

