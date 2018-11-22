import tensorflow as tf
from sklearn.datasets import load_digits

FLAGS = tf.app.flags.FLAGS


def main(_):
    """Main function for Domain Adaptation by Neural Networks - DANN"""
    # load the data
    digits = load_digits(return_X_y=True)
    # split into train and validation sets
    train_images = digits[0][:int(len(digits[0]) * 0.8)]
    train_labels = digits[1][:int(len(digits[0]) * 0.8)]
    valid_images = digits[0][int(len(digits[0]) * 0.8):]
    valid_labels = digits[1][int(len(digits[0]) * 0.8):]

    print(train_images)
    print(train_labels)


if __name__ == '__main__':
    tf.app.run()
