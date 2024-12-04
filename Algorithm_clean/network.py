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

import tensorflow as tf
from opts import opt

def get_crop_shape(target, refer):
    # depth, the 4th dimension
    cd = (target.get_shape()[3] - refer.get_shape()[3])
    assert (cd >= 0)
    if cd % 2 != 0:
        cd1, cd2 = int(cd//2), int(cd//2) + 1
    else:
        cd1, cd2 = int(cd//2), int(cd//2)
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw//2), int(cw//2) + 1
    else:
        cw1, cw2 = int(cw//2), int(cw//2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch//2), int(ch//2) + 1
    else:
        ch1, ch2 = int(ch//2), int(ch//2)

    return (ch1, ch2), (cw1, cw2), (cd1, cd2)

def get_unet(img_size=(256,256,256), num_channels=opt.num_channels, num_classes=opt.num_classes):
    
    concat_axis = 4
    inputs = tf.keras.layers.Input(shape=img_size+(num_channels,))
    conv1 = tf.keras.layers.Conv3D(8, (3, 3, 3), activation='relu', padding='same', name='conv1')(inputs)
    conv1 = tf.keras.layers.Conv3D(8, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    conv3 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    
    conv4 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv6 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv4)
    
    up_conv6 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(conv6)
    ch, cw, cd = get_crop_shape(conv3, up_conv6)
    crop_conv3 = tf.keras.layers.Cropping3D(cropping=(ch,cw,cd))(conv3)
    up7 = tf.keras.layers.concatenate([up_conv6, crop_conv3], axis=concat_axis) 
    conv7 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv7)
    
    up_conv7 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(conv7)
    ch, cw, cd = get_crop_shape(conv2, up_conv7)
    crop_conv2 = tf.keras.layers.Cropping3D(cropping=(ch,cw,cd))(conv2)
    up8 = tf.keras.layers.concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv8)
    
    up_conv8 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(conv8)
    ch, cw, cd = get_crop_shape(conv1, up_conv8)
    crop_conv1 = tf.keras.layers.Cropping3D(cropping=(ch,cw,cd))(conv1)
    up9 = tf.keras.layers.concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = tf.keras.layers.Conv3D(8, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = tf.keras.layers.Conv3D(8, (3, 3, 3), activation='relu', padding='same')(conv9)
    
    ch, cw, cd = get_crop_shape(inputs, conv9)
    conv9 = tf.keras.layers.ZeroPadding3D(padding=((ch[0], ch[1]), (cw[0], cw[1]), (cd[0], cd[1])))(conv9)
    conv10 = tf.keras.layers.Conv3D(num_classes, (1, 1, 1), activation="softmax")(conv9)
    
    model = tf.keras.Model(inputs=inputs, outputs = conv10)
    # model.summary(line_length=200)
    
    # tf.keras.utils.plot_model(model, "UNet_aorta_seg.png", show_shapes=True)
    
    return model
