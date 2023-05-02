# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:26:31 2023

@author: vrieshs
"""

from radiomics import featureextractor
from collections import defaultdict
from collections import OrderedDict
import pandas as pd
import os
from opts import opt
import SimpleITK as sitk


def radiomics_feature_extraction(image, label, output_dir=opt.data_dir):
    '''
    PyRadiomics feature extraction

    '''

    #Settings
    settings = {}
    settings['binWidth'] = opt.radiomics_binwidth
    settings['resampledPixelSpacing'] = opt.radiomics_pixelspacing
    settings['interpolator'] = opt.radiomics_interpolator
    settings['verbose'] = opt.radiomics_verbose

    features_all2 = defaultdict(list)

    #Extract features
    featureVector = {}
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    if opt.segment_voxel == 'segment':
        featureVector = extractor.execute(image, label)
        for k, v in featureVector.items():
            features_all[k].append(v)
            features_all2[k].append(v)
    elif opt.segment_voxel == 'voxel':
        featureVector = extractor.execute(image, label, voxelBased=True)
        for k, v in featureVector.items():
            if isinstance(v, sitk.Image):
                sitk.WriteImage(v, os.path.join(output_dir, k +'.nrrd'), True)
            else:
                features_all[k].append(v)
                features_all2[k].append(v)
    #for f in featureVector.keys(): # uncomment if you want the features + values printed
        #print ('%s: %s' %(f, featureVector[f]))

    features_all = {x:[] for x in featureVector.keys()}

    return pd.DataFrame.from_dict(features_all2)