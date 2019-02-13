import tensorflow as tf
import pickle as pkl
import os
import numpy as np
'''
!!! RIGHT NOW WITHOUT NORMALIZING DATA (MEAN)! !!!
'''

RGB2GRAY = [0.2126, 0.7152, 0.0722]  # [0.2989, 0.5870, 0.1140]


def load_mnist(channel_size=3, truncate=False):
    # Get mnist
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize
    x_train, x_test = x_train / 255.0, x_test / 255.0
    if truncate:
        x_train = x_train[:-999]
        y_train = y_train[:-999]
        x_test = x_test[:-999]
        y_test = y_test[:-999]
    # MNIST only has 1 channel, so we gotta change it to the desired channel size (RGB 3 channels
    # or grayscale 1 channel)
    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    x_test = np.reshape(x_test, (-1, 28, 28, 1))
    if channel_size == 3:
        x_train = np.concatenate([x_train, x_train, x_train], axis=3)
        x_test = np.concatenate([x_test, x_test, x_test], axis=3)
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
    return (x_train, y_train), (x_test, y_test)


def load_mnistm(channel_size=3):
    # Load mnistm pkl
    mnistm = pkl.load(open(os.path.join("..", "datasets", "mnist-m", "mnistm_data.pkl"), 'rb'))
    # Get mnistm
    x_train_m, y_train_m, x_test_m, y_test_m = mnistm['X_train'], mnistm['Y_train'], mnistm['X_test'], mnistm['Y_test']
    # Normalize
    x_train_m, x_test_m = x_train_m / 255.0, x_test_m / 255.0
    if channel_size == 1:
        # Turn RGB images into grayscale
        x_train_m = np.multiply(x_train_m, RGB2GRAY)
        x_train_m = np.sum(x_train_m, axis=3)
        x_test_m = np.multiply(x_test_m, RGB2GRAY)
        x_test_m = np.sum(x_test_m, axis=3)
        x_train_m = np.reshape(x_train_m, (-1, 32, 32, 1))
        x_test_m = np.reshape(x_test_m, (-1, 32, 32, 1))
    return (x_train_m, y_train_m), (x_test_m, y_test_m)


def load_svhn(channel_size=3, truncate=True):
    # load svhn
    svhn = pkl.load(open(os.path.join("..", "datasets", "svhn", "svhn_data.pkl"), 'rb'))
    # Get svhn
    x_train, y_train, x_test, y_test = svhn['x_train'], svhn['y_train'], svhn['x_test'], svhn['y_test']
    # Normalize
    x_train, x_test = x_train / 255.0, x_test / 255.0
    if channel_size == 1:
        # Turn RGB images into grayscale
        x_train = np.multiply(x_train, RGB2GRAY)
        x_train = np.sum(x_train, axis=3)
        x_test = np.multiply(x_test, RGB2GRAY)
        x_test = np.sum(x_test, axis=3)
        x_train = np.reshape(x_train, (-1, 32, 32, 1))
        x_test = np.reshape(x_test, (-1, 32, 32, 1))
    if truncate:
        inds_train = np.random.permutation(len(x_train))[:60000]
        inds_test = np.random.permutation(len(x_test))[:10000]
        inds_train.sort()
        inds_test.sort()
        x_train = x_train[inds_train]
        y_train = y_train[inds_train]
        x_test = x_test[inds_test]
        y_test = y_test[inds_test]
    print(x_train.shape)
    print(x_test.shape)
    return (x_train, y_train), (x_test, y_test)


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def zero_masked_layer(input_layer, level):
    mask = tf.keras.backend.random_binomial(shape=tf.shape(input_layer), p=level, dtype=input_layer.dtype)
    return tf.multiply(input_layer, mask)


def random_rotate(input_layer, max_degree):
    angle = np.random.randint(0, max_degree+1)
    return tf.contrib.image.rotate(input_layer, angle)


def random_translate(input_layer, max_dx, max_dy, size):
    dx = np.random.randint(-max_dx, max_dx+1, size=size)
    dy = np.random.randint(-max_dy, max_dy+1, size=size)
    translations = list(map(list, zip(dx, dy)))
    return tf.contrib.image.translate(input_layer, translations=translations)


def augment_data(input_layer, max_degree=20, max_dx=6, max_dy=6, size=128):
    input_layer = random_translate(input_layer, max_dx, max_dy, size)
    return random_rotate(input_layer, max_degree)


def add_noise(input_layer, noise='zero', level=0.5):
    if noise == 'zero':
        return zero_masked_layer(input_layer, level)
    else:
        return gaussian_noise_layer(input_layer, level)


if __name__ == '__main__':
    load_svhn(channel_size=1)
