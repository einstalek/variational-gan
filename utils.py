import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from skimage.transform import resize


def get_data_generator(rotate=20, 
                       horizontal_flip=True,
                       width_shift_range=0.15, 
                       height_shift_range=0.15,
                       zoom_range=0.15
                      ):
    """
    Creates instance of tensorflow Data Generator
    """
    return ImageDataGenerator(
        rotation_range=rotate,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip, 
        zoom_range=zoom_range
    )


def name(n):
    """
    Get image name by its index in celebA dataset
    """
    return "%.6d" % (n+1) + ".jpg"


def pic(dirp, n, w=224, h=224):
    """
    Loads image with id from image directory
    
    :param dirp: path to images
    :param n: image index 
    :param w: width to resize image to
    :param h: height to resize image to
    """
    img = plt.imread(os.path.join(dirp, name(n)))
    img = resize(img, (h, w))
    return img


class DataGenerator(tf.keras.utils.Sequence):
    """
    Data generator class, used for training 
    Returns original and distorted images
    """
    def __init__(self,
                 dirp, 
                 batch_size=8, 
                 rotate=20, 
                 horizontal_flip=True,
                 width_shift_range=0.15, 
                 height_shift_range=0.15,
                 zoom_range=0.15, 
                 noise_max=0.1):
        """
        :param batch_size: batch size
        :dirp: path to images directory
        """
        self.batch_size = batch_size
        self.ids = np.arange(200_000)
        self.dirp = dirp
        self.noise_max = noise_max
        self.img_gen = get_data_generator(rotate=rotate, 
                                          horizontal_flip=horizontal_flip, 
                                          width_shift_range=width_shift_range, 
                                          height_shift_range=height_shift_range,
                                          zoom_range=zoom_range)
        self.on_epoch_end()
        
    def __getitem__(self, index):
        ids = self.ids[self.batch_size * index : self.batch_size * (index + 1)]
        # original images
        X_origin = np.empty((self.batch_size, *img.shape), dtype=np.float32)
        # distorted images
        X_distored = np.empty((self.batch_size, *img.shape), dtype=np.float32)
        for i, _id in enumerate(ids):
            X_origin[i] = pic(self.dirp, _id)
            # apply distortion to original images
            X_distored[i] = self.img_gen.random_transform(X_origin[i])
        # add noise to distorted images
        X_distored += np.random.normal(0, self.noise_max, size=X_distored.shape)
        return X_origin, X_distored
    
    def __len__(self):
        return len(self.ids) // self.batch_size
    
    def on_epoch_end(self):
        np.random.shuffle(self.ids)

