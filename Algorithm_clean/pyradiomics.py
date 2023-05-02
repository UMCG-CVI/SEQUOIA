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