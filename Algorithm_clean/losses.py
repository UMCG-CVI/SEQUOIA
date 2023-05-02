# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 10:19:43 2021

@author: PraaghGd
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.utils.extmath import cartesian
from opts import opt
from scipy.spatial.distance import directed_hausdorff


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred):
    numlabels = opt.num_classes
    dice = 0
    for i in range(numlabels):
        dice += dice_coef(y_true[...,i], y_pred[...,i])
    return dice/numlabels

def dice_coef_loss(y_true, y_pred):
    return -dice_coef_multilabel(y_true, y_pred)
