# -*- coding: utf-8 -*-
"""
SEQUOIA - automated multiclass SEgmentation, QUantification, and visualizatiOn of the dIseased Aorta on hybrid PET/CT

@author: praaghgd
"""

import SimpleITK as sitk    


class Crop(object):
    """
    Randomly crop a window of given shape out of a bigger array.
    Array must have at least three dimensions.

    Parameters:
    img: image to crop
    seg: another array to crop in the same way (optional)
    shape: shape of window to crop (tuple of 3 integers)
    ignore_first_axis: if true, first axis will be ignored, and the entire
    batch will be cropped according to the next 3 axes
    """

    def __init__(self, output_size, index=[]):
        self.name = 'Crop2'

        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

        assert isinstance(index, (int, tuple, list))
        if isinstance(index, int):
            self.index = (index, index, index)
        else:
            # assert len(index) == 3
            self.index = index

    def __call__(self, sample, j):
        image = sample['image']
        
        # guarantee label type to be integer
        roiFilter = sitk.RegionOfInterestImageFilter()
        roiFilter.SetSize(self.output_size)
        roiFilter.SetIndex(self.index[j])

        for channel in range(len(image)):
            image[channel] = roiFilter.Execute(image[channel])

        return {'image': image}



