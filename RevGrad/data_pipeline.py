import tensorflow as tf
import pickle as pkl
import os
import numpy as np
'''
!!! RIGHT NOW WITHOUT NORMALIZING DATA (MEAN)! !!!
'''

def load_data():
    # load the data
    mnist = tf.keras.datasets.mnist
    mnistm = pkl.load(open(os.path.join("..", "datasets", "mnist-m", "mnistm_data.pkl"), 'rb'))
    # Get mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # MNIST dataset contains 60000 sets, MNIST-M only 59001.
    # We also wanna change the width and height to 32 pixels.
    x_train = np.pad(X_train[:-999], [(0, 0), (2, 2), (2, 2)], mode='constant')
    y_train = y_train[:-999]
    x_test = np.pad(X_test[:-999], [(0, 0), (2, 2), (2, 2)], mode='constant')
    y_test = y_test[:-999]
    # # Normalize
    # x_train, x_test = x_train / 255.0, x_test / 255.0

    # Get mnistm
    x_train_m, y_train_m, x_test_m, y_test_m = mnistm['X_train'], mnistm['Y_train'], mnistm['X_test'], mnistm['Y_train']
    # Normalize
    # x_train_m, x_test_m = x_train_m / 255.0, x_test_m / 255.0

    # MNIST only has 1 channel, so we gotta change it to RGB (3 channels)
    x_train = np.reshape(x_train, (-1, 32, 32, 1))
    x_test = np.reshape(x_test, (-1, 32, 32, 1))
    x_train = np.concatenate([x_train, x_train, x_train], 3)
    x_test = np.concatenate([x_test, x_test, x_test], 3)

    return (x_train, y_train), (x_test, y_test), (x_train_m, y_train_m), (x_test_m, y_test_m)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(60000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


if __name__ == '__main__':
    load_data()