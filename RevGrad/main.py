import tensorflow as tf
from tensorflow.python.framework import ops
import data_pipeline as dp
import math
import utilities
import random


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 128, 'Batch size. ')
tf.flags.DEFINE_integer('total_epochs', 1000, 'Number of epochs for training')
tf.flags.DEFINE_float('base_learning_rate', 0.01, 'Base learning rate for learning rate decay')

tf.logging.set_verbosity(tf.logging.DEBUG)


def estimator_model_fn(features, labels, mode, params):
    """The estimator function"""
    features['x_s'] = tf.Print(features['x_s'], [tf.shape(features['x_s'])], "Features shape..")
    input_layer_source = tf.feature_column.input_layer({"x_s": features['x_s']}, params['feature_columns'][0])
    input_layer_target = tf.feature_column.input_layer({"x_t": features['x_t']}, params['feature_columns'][1])
    input_layer_source = tf.Print(input_layer_source, [tf.shape(input_layer_source)], "Input layer source shape")
    # CNNs need input data to be of shape [batch_size, width, height, channel]
    input_layer_source = tf.reshape(input_layer_source, [128, 32, 32, 3])
    input_layer_target = tf.reshape(input_layer_target, [128, 32, 32, 3])
    input_layer_source = tf.Print(input_layer_source, [tf.shape(input_layer_source)], "Input layer source shape after")

    if mode == tf.estimator.ModeKeys.PREDICT:
        # To be implemented
        return

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Gotta change labels to one-hot
        labels = tf.Print(labels, [tf.shape(labels), tf.shape(input_layer_source)], 'Labels shape')
        class_labels = tf.one_hot(labels, 10)
        domain_labels_source = [0] * 128  # tf.shape(input_layer_source)[0]
        domain_labels_source = tf.one_hot(domain_labels_source, 2)
        domain_labels_target = [1] * 128  # tf.shape(input_layer_target)[0]
        domain_labels_target = tf.one_hot(domain_labels_target, 2)
        # Apply DANN model to both input layers
        # TODO !!! CALCULATE CORRECT ALPHA VALUE FOR REVGRAD!
        class_logits_source, domain_logits_source, s1 = dann_model_fn(input_layer_source, is_training=True)
        tf.identity(tf.shape(s1), 's1')
        _, domain_logits_target, s2 = dann_model_fn(input_layer_target, is_training=True)
        tf.identity(tf.shape(s2), 's2')
        class_loss = tf.losses.softmax_cross_entropy(class_labels, logits=class_logits_source)
        domain_loss_source = tf.losses.softmax_cross_entropy(domain_labels_source, logits=domain_logits_source)
        domain_loss_target = tf.losses.softmax_cross_entropy(domain_labels_target, logits=domain_logits_target)
        total_loss = tf.reduce_mean(class_loss) + tf.reduce_mean(domain_loss_source) + tf.reduce_mean(domain_loss_target)
        learning_rate = utilities.lr_annealing(learning_rate=FLAGS.base_learning_rate,
                                               global_step=tf.train.get_global_step(),
                                               alpha=0.001,
                                               beta=0.75)
        global_step = tf.train.get_global_step()
        tf.identity(learning_rate, 'learning_rate')
        tf.identity(global_step, 'global_step')
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def dann_model_fn(input_layer, alpha=1.0, is_training=False):
    """The actual DANN model architecture"""
    with tf.variable_scope('feature_extractor', reuse=tf.AUTO_REUSE):
        out = tf.layers.conv2d(input_layer, filters=64, kernel_size=5)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.layers.max_pooling2d(out, 2, 2)
        out = tf.nn.relu(out)
        out = tf.layers.conv2d(out, filters=50, kernel_size=5)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.layers.dropout(out, training=is_training)
        out = tf.layers.max_pooling2d(out, 2, 2)
        out = tf.nn.relu(out)
        out = tf.layers.flatten(out)
        out = tf.Print(out, [tf.shape(out), tf.shape(input_layer)], 'out tensor after flatten')
    with tf.variable_scope('class_classifier', reuse=tf.AUTO_REUSE):
        class_out = tf.layers.dense(out, 100)
        class_out = tf.layers.batch_normalization(class_out, training=is_training)
        class_out = tf.nn.relu(class_out)
        class_out = tf.layers.dropout(class_out, training=is_training)
        class_out = tf.layers.dense(class_out, 100)
        class_out = tf.layers.batch_normalization(class_out, training=is_training)
        class_out = tf.nn.relu(class_out)
        class_logits = tf.layers.dense(class_out, 10, name='class_logit')
    with tf.variable_scope('domain_classifier', reuse=tf.AUTO_REUSE):
        # Flip gradient when domain classifier is backpropagated
        grad_name = "RevGrad%d" % random.randint(1, 1000)
        @ops.RegisterGradient(grad_name)
        def _reverse_gradient(op, grad):
            return [tf.negative(grad) * alpha]
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            domain_out = tf.identity(out)
        s = domain_out
        domain_out = tf.layers.dense(domain_out, 100)
        domain_out = tf.layers.batch_normalization(domain_out, training=is_training)
        domain_out = tf.nn.relu(domain_out)
        domain_logits = tf.layers.dense(domain_out, 2)

    return class_logits, domain_logits, s


def main(_):
    """Main function for Domain Adaptation by Neural Networks - DANN"""
    tf.reset_default_graph()
    # Load MNIST and MNIST-M dataset
    (x_train, y_train), (x_test, y_test), (x_m_train, y_m_train), (x_m_test, y_m_test) = dp.load_data()

    # Configurations first
    total_steps_training = math.ceil((x_train.shape[0] / FLAGS.batch_size) * FLAGS.total_epochs)
    print(total_steps_training)

    # We are working with transformed MNIST dataset => image shape is 32x32x3
    feature_columns = [tf.feature_column.numeric_column("x_s", shape=(32, 32, 3)),
                       tf.feature_column.numeric_column("x_t", shape=(32, 32, 3))]

    # Set up the estimator
    classifier = tf.estimator.Estimator(
        model_fn=estimator_model_fn,
        model_dir="./model",
        params={
            'feature_columns': feature_columns,
        }
    )
    # Set up logging in training mode
    logging_hook = tf.train.LoggingTensorHook(
        tensors={"lr": "learning_rate", "gs": "global_step", "s1": "s1", "s2": "s2"},
        every_n_secs=1)
    # Train DANN
    classifier.train(
        # input_fn=lambda: dp.train_input_fn({'x_s': x_train, 'x_t': x_m_train}, y_train, FLAGS.batch_size),
        input_fn=tf.estimator.inputs.numpy_input_fn({'x_s': x_train, 'x_t': x_m_train}, y_train, shuffle=True,
                                                    batch_size=128),
        max_steps=total_steps_training,
        hooks=[logging_hook]
    )


if __name__ == '__main__':
    tf.app.run()
