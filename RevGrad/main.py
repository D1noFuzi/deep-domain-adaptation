import tensorflow as tf
import data_pipeline as dp


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size. ')


def estimator_model_fn(features, labels, mode, params):
    """The estimator function"""

    input_layer_source = tf.feature_column.input_layer({"x_s": features['x_s']}, params['feature_columns'][0])
    input_layer_target = tf.feature_column.input_layer({"x_t": features['x_t']}, params['feature_columns'][1])
    # CNNs need input data to be of shape [batch_size, width, height, channel]
    input_layer_source = tf.reshape(input_layer_source, [-1, 32, 32, 3])
    input_layer_target = tf.reshape(input_layer_target, [-1, 32, 32, 3])
    # Gotta change labels to one-hot
    labels = tf.one_hot(labels, 10)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Apply DANN model to both input layers
        class_logits_source, domain_logits_source = dann_model_fn(input_layer_source, is_training=True)
        _, domain_logits_target = dann_model_fn(input_layer_target, is_training=True)

def dann_model_fn(input_layer, is_training):
    """The actual DANN model architecture"""
    with tf.variable_scope('feature_extractor'):
        out = tf.layers.conv2d(input_layer, filters=64, kernel_size=5)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.layers.max_pooling2d(out, 2)
        out = tf.nn.relu(out)
        out = tf.layers.conv2d(out, filters=50, kernel_size=5)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.layers.dropout(out)
        out = tf.layers.max_pooling2d(out, 2)
        out = tf.nn.relu(out)
    with tf.variable_scope('class_classifier'):
        class_out = tf.layers.dense(out, 100)
        class_out = tf.layers.batch_normalization(class_out, training=is_training)
        class_out = tf.nn.relu(class_out)
        class_out = tf.layers.dropout(class_out)
        class_out = tf.layers.dense(class_out, 100)
        class_out = tf.layers.batch_normalization(class_out, training=is_training)
        class_out = tf.nn.relu(class_out)
        class_logits = tf.layers.dense(class_out, 10)
    with tf.variable_scope('domain_classifier'):
        domain_out = tf.layers.dense(out, 100)
        domain_out = tf.layers.batch_normalization(domain_out, training=is_training)
        domain_out = tf.nn.relu(domain_out)
        domain_logits = tf.layers.dense(domain_out, 2)

    return class_logits, domain_logits

def main(_):
    """Main function for Domain Adaptation by Neural Networks - DANN"""
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test), (x_m_train, y_m_train), (x_m_test, y_m_test) = dp.load_data()
    # We are working with transformed MNIST dataset => image shape is 32x32x3
    feature_columns = [tf.feature_column.numeric_column("x_s", shape=[32, 32, 3]),
                       tf.feature_column.numeric_column("x_t", shape=[32, 32, 3])]

    # Set up the estimator
    classifier = tf.estimator.Estimator(
        model_fn=estimator_model_fn,
        params={
            'feature_columns': feature_columns
        }
    )
    # Train DANN
    classifier.train(
        input_fn=lambda: dp.train_input_fn({'x_s': x_train, 'x_t': x_m_train}, y_train, FLAGS.batch_size)
    )


if __name__ == '__main__':
    tf.app.run()
