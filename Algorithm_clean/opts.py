# -*- coding: utf-8 -*-
"""
SEQUOIA - automated multiclass SEgmentation, QUantification, and visualizatiOn of the dIseased Aorta on hybrid PET/CT

@author: praaghgd
"""

import argparse

parser = argparse.ArgumentParser(description='Define prediction values of the aorta analyzation software')
parser.add_argument('--data_dir', type=str, default=r'...', help='where dataset saved. Within this folder, save every patient as different folder, with ct and pet images in that folder')
parser.add_argument('--ct_filename', type=str, default='ct.nii.gz', help='filenames of the ct image')
parser.add_argument('--pet_filename', type=str, default='pet.nii.gz', help='filenames of the pet image, choose "none" if no PET image is available')
parser.add_argument('--speed', type=str, default='accurate', help='choose "accurate", "medium", or "fast" (slower, but more accurate to less accurate, but quicker)')
parser.add_argument('--calcium_threshold', type=str, default='standard', help='choose "standard" for 130 HU, "100kVp" to use 147HU for 100 kVp (Nakazato et al. JCCT 2009), or "SD" for mean background + 3 * SD background (Raggi et al. AJR 2002)')
parser.add_argument('--per_calc', type=str, default=False, help='choose True if you want an analysis per calcification')
parser.add_argument('--pet_threshold', type=str, default='A50P', help='choose "number" for that specific SUV or "A50P" for the background adapted 50% SUV peak threshold')
parser.add_argument('--meshes', type=bool, default=True, help='choose True if you want to get a heatmap of the calcium and pet hotspot distribution')

parser.add_argument('--radiomics', type=bool, default=False, help='choose True if you do want the radiomics features extracted')
parser.add_argument('--radiomics_binwidth', type=float, default=0.5, help='choose your own bin width. 0.5 has shown to work well for PET imaging')
parser.add_argument('--radiomics_pixelspacing', type=list, default=[2,2,2], help='choose pixelspacing you want it to resample it to')
parser.add_argument('--radiomics_interpolator', type=str, default='sitkBSpline', help='choose the interpolation method')
parser.add_argument('--radiomics_verbose', type=bool, default=True, help='choose True or False')
parser.add_argument('--segment_voxel', type=str, default='segment', help='choose "segment" for segment based extraction or "voxel" for voxel based extraction')

# DO NOT change these arguments
parser.add_argument('--checkpoint_path', type=str, default=r'checkpoint', help='where checkpoint is saved')
parser.add_argument('--batch_size', type=int, default=10, help='input batch size')
parser.add_argument('--img_size', nargs='+', type=int, default=(160,160,64), help='wanted input size of the image')
parser.add_argument('--num_channels', type=int, default=1, help='# of channels')
parser.add_argument('--num_classes', type=int, default=5, help='# of classes')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')

opt = parser.parse_args()

args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')
