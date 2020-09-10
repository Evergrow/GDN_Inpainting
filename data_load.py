import os
import glob
import scipy
import random
import numpy as np
import tensorflow as tf

class Dataset(object):
    def __init__(self, config):
        super(Dataset, self).__init__()

        self.train_list = self.load_flist(config.TRAIN_FLIST)
        self.mask_list = self.load_flist(config.MASK_FLIST)
        self.val_list = self.load_flist(config.VAL_FLIST)
        self.mask_val_list = self.load_flist(config.VAL_MASK_FLIST)
        self.len_train = len(self.train_list)
        self.len_val = len(self.val_list)
        self.input_size = config.INPUT_SIZE
        self.epoch = config.EPOCH
        self.batch_size = config.BATCH_SIZE
        self.val_batch_size = config.VAL_BATCH_SIZE
        self.data_batch()

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist
        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist
            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []

    def data_batch(self):
        train_image = tf.data.Dataset.from_tensor_slices(self.train_list)
        train_mask = tf.data.Dataset.from_tensor_slices(self.mask_list)
        val_image = tf.data.Dataset.from_tensor_slices(self.val_list)
        val_mask = tf.data.Dataset.from_tensor_slices(self.mask_val_list)

        def image_fn(img_path):
            x = tf.read_file(img_path)
            x_decode = tf.image.decode_jpeg(x, channels=3)
            img = tf.image.resize_images(x_decode, [self.input_size, self.input_size])
            # img = tf.random_crop(x_decode, [self.input_size, self.input_size, 3])
            img = tf.cast(img, tf.float32)
            return img

        def mask_fn(mask_path):
            x = tf.read_file(mask_path)
            x_decode = tf.image.decode_jpeg(x, channels=1)
            mask = tf.image.resize_images(x_decode, [self.input_size, self.input_size])
            mask = tf.cast(mask, tf.float32)
            return mask

        train_image = train_image. \
            repeat(self.epoch). \
            map(image_fn). \
            apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)). \
            shuffle(buffer_size=1000)

        train_mask = train_mask. \
            repeat(self.epoch). \
            map(mask_fn). \
            apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)). \
            shuffle(buffer_size=1000)

        val_image = val_image. \
            repeat(10 * self.epoch). \
            map(image_fn). \
            apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))

        val_mask = val_mask. \
            repeat(10 * self.epoch). \
            map(mask_fn). \
            apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))

        self.batch_image = train_image.make_one_shot_iterator().get_next()
        self.batch_mask = train_mask.make_one_shot_iterator().get_next()
        self.val_image = val_image.make_one_shot_iterator().get_next()
        self.val_mask = val_mask.make_one_shot_iterator().get_next()

