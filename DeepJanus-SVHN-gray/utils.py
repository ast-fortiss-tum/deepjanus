from os import makedirs
from os.path import exists, basename, join

from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from folder import Folder
from config import IMG_SIZE, IMG_CHN
import numpy as np

# load the MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


def get_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def print_archive(archive):
    dst = Folder.DST_ARC + "_DJ"
    if not exists(dst):
        makedirs(dst)
    for i, ind in enumerate(archive):
        filename1 = join(dst, basename(
            'archived_' + str(i) +
            '_mem1_l_' + str(ind.m1.predicted_label) +
            '_seed_' + str(ind.seed)))
        if IMG_CHN == 1:
            plt.imsave(filename1, ind.m1.purified.reshape(IMG_SIZE, IMG_SIZE),
                       cmap=cm.gray,
                       format='png')
        else:
            plt.imsave(filename1, ind.m1.purified.reshape(IMG_SIZE, IMG_SIZE, IMG_CHN),
                       cmap=cm.gray,
                       format='png')
        np.save(filename1, ind.m1.purified)
        assert (np.array_equal(ind.m1.purified,
                               np.load(filename1 + '.npy')))

        filename2 = join(dst, basename(
            'archived_' + str(i) +
            '_mem2_l_' + str(ind.m2.predicted_label) +
            '_seed_' + str(ind.seed)))
        if IMG_CHN == 1:
            plt.imsave(filename2, ind.m2.purified.reshape(IMG_SIZE, IMG_SIZE),
                       cmap=cm.gray,
                       format='png')
        else:
            plt.imsave(filename2, ind.m2.purified.reshape(IMG_SIZE, IMG_SIZE, IMG_CHN),
                       cmap=cm.gray,
                       format='png')
        np.save(filename2, ind.m2.purified)
        assert (np.array_equal(ind.m2.purified,
                               np.load(filename2 + '.npy')))


def print_archive_experiment(archive):
    for i, ind in enumerate(archive):
        digit = ind.m1
        digit.export(ind.id)
        digit = ind.m2
        digit.export(ind.id)
        ind.export()


# Useful function that shapes the input in the format accepted by the ML model.
def reshape(v):
    v = (np.expand_dims(v, 0))
    # Shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        v = v.reshape(v.shape[0], 1, IMG_SIZE, IMG_SIZE)
    else:
        v = v.reshape(v.shape[0], IMG_SIZE, IMG_SIZE, 1)
    v = v.astype('float32')
    v = v / 255.0
    return v


def input_reshape(x):
    # shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        x_reshape = x.reshape(x.shape[0], 1, 32, 32)
    else:
        x_reshape = x.reshape(x.shape[0], 32, 32, 1)
    x_reshape = x_reshape.astype('float32')
    x_reshape /= 255.0
    return x_reshape

def set_all_seeds(seed):
    import random
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)