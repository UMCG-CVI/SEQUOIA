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

import numpy as np
import tensorflow as tf
import SimpleITK as sitk
import copy


class GetDataset(tf.keras.utils.Sequence):
    """
    load image-label pair for training, testing and inference.
    Currently only support linear interpolation method
    Args:
        data_dir (string): Path to data directory.
        image_filename (string): Filename of image data.
        transforms (list): List of SimpleITK image transformations.
        train (bool): Determine whether the dataset class run in training/inference mode. When set to false, an empty label with same metadata as image is generated.
    """

    def __init__(self, sample=[], input_path=[], batch_size=10, img_size=(160,160,64), num_channels=1, transforms=None):

        # Init membership variables
        self.sample = sample
        self.input_path = input_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_channels = num_channels
        self.transforms = transforms
    
    def __len__(self):
        return (len(self.input_path) * self.batch_size) // self.batch_size

    def __getitem__(self,idx):
        i = idx * self.batch_size
        x = np.zeros((self.batch_size,) + self.img_size + (self.num_channels,), dtype="float32")
        new_input_path = [path for path in self.input_path for b in range(self.batch_size)]
        batch_cases = new_input_path[i : i + self.batch_size]
        samples = [copy.deepcopy(self.sample) for x in range(self.batch_size)]
        for j, path in enumerate(batch_cases):
            image_np = self.input_parser(samples[j], path, j)
            x[j] = image_np
        
        return x
    
        
    def input_parser(self, sample, path, j):
        sample_backup = copy.deepcopy(sample)
        if self.transforms:
            for transform in self.transforms:
                try:
                    sample = transform(sample, j)
                except:
                    print("Dataset preprocessing error: {} transform: {}".format(path.split('\\')[-1],transform.name))
                    sample = transform(sample, j)
                    exit()
        
        # convert sample to tf tensors
        for channel in range(len(sample['image'])):
            image_np_ = sitk.GetArrayFromImage(sample['image'][channel])
            image_np_ = np.asarray(image_np_,np.float32)
            # to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])
            image_np_ = np.transpose(image_np_,(1,2,0))
            if channel == 0:
                image_np = image_np_[:,:,:,np.newaxis]
            else:
                image_np = np.append(image_np,image_np_[:,:,:,np.newaxis],axis=-1)
        
        if np.any(np.isnan(image_np)):
            c = 0
            while np.any(np.isnan(image_np)):
                sample = copy.deepcopy(sample_backup)
                if self.transforms:
                    for transform in self.transforms:
                        try:
                            sample = transform(sample, j)
                        except:
                            print("Dataset preprocessing error: {} transform: {}".format(path.split('\\')[-1],transform.name))
                            sample = transform(sample, j)
                            exit()
                
                # convert sample to tf tensors
                for channel in range(len(sample['image'])):
                    image_np_ = sitk.GetArrayFromImage(sample['image'][channel])
                    image_np_ = np.asarray(image_np_,np.float32)
                    # to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])
                    image_np_ = np.transpose(image_np_,(1,2,0))
                    if channel == 0:
                        image_np = image_np_[:,:,:,np.newaxis]
                    else:
                        image_np = np.append(image_np,image_np_[:,:,:,np.newaxis],axis=-1)
                c += 1
                
                if c > 3 and np.any(np.isnan(image_np)):
                    image_np[np.isnan(image_np)] = 0

        return image_np






