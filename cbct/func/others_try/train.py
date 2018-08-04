import threading
import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split

from func.data_io import DataSet
from func.others_try.model import get_dilated_unet
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os

from func.train_model import get_learning_rate_scheduler

BATCH_SIZE = 6


class ThreadSafeIterator:

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*args, **kwargs):
        return ThreadSafeIterator(f(*args, **kwargs))

    return g


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    KTF.set_session(sess)

    h5_data_path = "/home/jzhang/helloworld/mtcnn/cb/inputs/data_0717.hdf5"
    fold_k = "01"
    random_k_fold = False
    input_channels = 1
    output_channels = 1
    random_crop_size = (512, 512)
    mask_nb = 0
    batch_size = BATCH_SIZE
    seed = 100

    train_ids = ['00000041', '00000042', '00000043', '00000044', '00000045', '00000046', '00000047', '00000048',
                 '00000049', '00000050', '00000051', '00000052', '00000053', '00000054', '00000055', '00000056',
                 '00000057', '00000058', '00000059', '00000060', '00000061', '00000062', '00000063', '00000064',
                 '00000065', '00000066', '00000067', '00000068', '00000069', '00000070', '00000071', '00000072',
                 '00000073', '00000074', '00000075', '00000076', '00000077', '00000078', '00000079', '00000080']

    val_ids = ['00000041', '00000059', '00000074', '00000075']
    # prepare train and val data.
    dataset = DataSet(h5_data_path, val_fold_nb=fold_k, random_k_fold=random_k_fold,
                      input_channels=input_channels, output_channels=output_channels,
                      random_crop_size=random_crop_size, mask_nb=mask_nb, batch_size=batch_size,
                      train_ids=train_ids,
                      val_ids=val_ids

                      )
    # we create two instances with the same arguments
    train_data_gen_args = dict(featurewise_center=False,
                               featurewise_std_normalization=False,
                               rotation_range=0,
                               width_shift_range=0.,
                               height_shift_range=0.,
                               horizontal_flip=True,
                               fill_mode="nearest",
                               shear_range=0.,
                               zoom_range=[1, 1.4],
                               )
    if random_crop_size is None:
        train_data_generator = dataset.get_keras_data_generator(is_train=True,
                                                                keras_data_gen_param=train_data_gen_args,
                                                                seed=seed)
    else:
        train_data_generator = dataset.get_custom_data_generator(is_train=True,
                                                                 keras_data_gen_param=train_data_gen_args,
                                                                 seed=seed)
    # no val augmentation.
    val_data_gen_args = dict(featurewise_center=False,
                             featurewise_std_normalization=False,
                             rotation_range=0.,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             horizontal_flip=False)

    val_data_generator = dataset.get_keras_data_generator(is_train=False, keras_data_gen_param=val_data_gen_args,
                                                          seed=seed)

    model = get_dilated_unet(
        input_shape=(None, None, 1),
        mode='cascade',
        filters=32,
        n_class=1
    )
    # model.load_weights("/home/jzhang/helloworld/mtcnn/cb/func/others_try/model_weights.hdf5")

    callbacks = [
        LearningRateScheduler(schedule=get_learning_rate_scheduler, verbose=0),


        # EarlyStopping(monitor='val_dice_coef',
        #                        patience=10,
        #                        verbose=1,
        #                        min_delta=1e-4,
        #                        mode='max'),
        # ReduceLROnPlateau(monitor='val_dice_coef',
        #                   factor=0.2,
        #                   patience=5,
        #                   verbose=1,
        #                   epsilon=1e-4,
        #                   mode='max'),
        ModelCheckpoint(monitor='val_dice_coef',
                        filepath='model_weights.hdf5',
                        save_best_only=True,
                        save_weights_only=True,
                        mode='max')]
    train_steps = dataset.get_train_val_steps(is_train=True)
    val_steps = dataset.get_train_val_steps(is_train=False)
    print("Training ...")

    model.fit_generator(generator=train_data_generator,
                        steps_per_epoch=train_steps,
                        epochs=100,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=val_data_generator,
                        validation_steps=val_steps)
