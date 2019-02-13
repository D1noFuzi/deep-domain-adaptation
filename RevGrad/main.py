import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import data_pipeline as dp
import math
import sys
sys.path.insert(0, '../../utilities')
import utilities
import random


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 128, 'Batch size.')
tf.flags.DEFINE_integer('total_epochs', 100, 'Number of epochs for training.')
tf.flags.DEFINE_float('base_learning_rate', 0.01, 'Base learning rate for learning rate decay.')
tf.flags.DEFINE_string('mode', 'train', 'Run the model in "train", "eval" or "predict" mode.')
tf.flags.DEFINE_string('source', 'mnist', 'Either mnist, mnistm or svhn.')
tf.flags.DEFINE_string('target', 'mnistm', 'Either mnist, mnistm or svhn.')
tf.flags.DEFINE_integer('channel_size', 3, 'Either 1 or 3. Defaults to 3 for RevGrad.')
tf.flags.DEFINE_string('model_dir', './model', 'Where to save the model.')
tf.flags.DEFINE_string('truncate_mnist', 'True', 'Either true or false. Truncates the mnist set to have same length.')
tf.flags.DEFINE_string('truncate_svhn', 'True', 'Either true or false. Truncates the svhn set to have same length.')
tf.flags.DEFINE_string('source_only', 'False', 'If set to True, will only train on source classification.')

tf.logging.set_verbosity(tf.logging.DEBUG)


def estimator_model_fn(features, labels, mode, params):
    """The estimator function"""
    input_layer_source = tf.feature_column.input_layer({"x_s": features['x_s']}, params['feature_columns'][0])
    input_layer_target = tf.feature_column.input_layer({"x_t": features['x_t']}, params['feature_columns'][1])
    # Reshape
    input_layer_source = tf.reshape(input_layer_source, [-1, 28, 28, FLAGS.channel_size])
    input_layer_target = tf.reshape(input_layer_target, [-1, 28, 28, FLAGS.channel_size])

    y_s = tf.cast(labels['y_s'], tf.int32)
    y_t = tf.cast(labels['y_t'], tf.int32)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # TODO: To be implemented
        return
    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        training = mode == tf.estimator.ModeKeys.TRAIN
        iter_ratio = params['iter_ratio']
        current_epoch = math_ops.ceil(math_ops.divide(tf.train.get_global_step(), iter_ratio))
        alpha = utilities.reverse_gradient_weight(current_epoch, FLAGS.total_epochs, 10.)
        # Apply DANN model
        class_logits_source, domain_logits_source = dann_model_fn(input_layer_source, alpha=alpha, is_training=training)
        class_logits_target, domain_logits_target = dann_model_fn(input_layer_target, alpha=alpha, is_training=training)
        # Get predicitons for accuracy
        pred_classes_target = tf.argmax(class_logits_target, axis=1, output_type=tf.int32)
        pred_classes_source = tf.argmax(class_logits_source, axis=1, output_type=tf.int32)
        # Create domain labels
        domain_labels_source = tf.zeros([tf.shape(features['x_s'])[0]], dtype=tf.int32)
        domain_labels_target = tf.ones([tf.shape(features['x_t'])[0]], dtype=tf.int32)
        # Compute losses
        class_loss = tf.losses.sparse_softmax_cross_entropy(labels=y_s,
                                                            logits=class_logits_source)
        domain_loss_source = tf.losses.sparse_softmax_cross_entropy(labels=domain_labels_source,
                                                                    logits=domain_logits_source)
        if FLAGS.source_only.lower() == 'false':
            domain_loss_target = tf.losses.sparse_softmax_cross_entropy(labels=domain_labels_target,
                                                                        logits=domain_logits_target)
            total_loss = tf.reduce_mean(class_loss) + tf.reduce_mean(domain_loss_source) + \
                         tf.reduce_mean(domain_loss_target)
        else:
            total_loss = tf.reduce_mean(class_loss)

        if mode == tf.estimator.ModeKeys.EVAL:
            source_class_acc = tf.metrics.accuracy(labels=y_s,
                                                   predictions=pred_classes_source,
                                                   name='source_class_acc_op')
            target_class_acc = tf.metrics.accuracy(labels=y_t,
                                                   predictions=pred_classes_target,
                                                   name='target_class_acc_op')
            metrics = {'source_class_acc': source_class_acc,
                       'target_class_acc': target_class_acc}
            return tf.estimator.EstimatorSpec(
                mode, loss=total_loss, eval_metric_ops=metrics)

        # Calculate a non streaming (per batch) accuracy
        source_class_acc = utilities.non_streaming_accuracy(pred_classes_source, y_s)
        target_class_acc = utilities.non_streaming_accuracy(pred_classes_target, y_t)

        # Initialize learning rate
        learning_rate = utilities.lr_annealing(learning_rate=FLAGS.base_learning_rate,
                                               current_epoch=current_epoch,
                                               total_epochs=FLAGS.total_epochs,
                                               alpha=10,
                                               beta=0.75)
        tf.identity(learning_rate, 'learning_rate')
        tf.identity(alpha, 'alpha')
        tf.identity(total_loss, 'loss')
        tf.identity(source_class_acc, 'source_class_acc')
        tf.identity(target_class_acc, 'target_class_acc')
        # TensorBoard
        tf.summary.scalar('Train_source_acc', source_class_acc)
        tf.summary.scalar('Train_target_acc', target_class_acc)
        tf.summary.scalar('Learning_rate', learning_rate)
        tf.summary.scalar('Alpha', alpha)
        tf.summary.merge_all()

        # Optimize
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def dann_model_fn(input_layer, alpha=1.0, is_training=False):
    """The actual DANN model architecture"""
    with tf.variable_scope('feature_extractor', reuse=tf.AUTO_REUSE):
        # input_layer = tf.Print(input_layer, [tf.shape(input_layer)], "Input shape before flatten ")
        out = tf.layers.conv2d(input_layer, filters=64, kernel_size=5)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.layers.max_pooling2d(out, 2, 2)
        out = tf.nn.relu(out)
        out = tf.layers.conv2d(out, filters=50, kernel_size=5)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.layers.dropout(out, training=is_training)
        out = tf.layers.max_pooling2d(out, 2, 2)
        out = tf.nn.relu(out)
        # out = tf.Print(out, [tf.shape(out)], "Out shape before flatten ")
        out = tf.layers.flatten(out)
    with tf.variable_scope('class_classifier', reuse=tf.AUTO_REUSE):
        class_out = tf.layers.dense(out, 100)
        class_out = tf.layers.batch_normalization(class_out, training=is_training)
        class_out = tf.nn.relu(class_out)
        class_out = tf.layers.dropout(class_out, training=is_training)
        class_out = tf.layers.dense(class_out, 100)
        class_out = tf.layers.batch_normalization(class_out, training=is_training)
        class_out = tf.nn.relu(class_out)
        class_logits = tf.layers.dense(class_out, units=10, name='class_logit')
    with tf.variable_scope('domain_classifier', reuse=tf.AUTO_REUSE):
        # Flip gradient when domain classifier is backpropagated
        grad_name = "RevGrad%d" % random.randint(1, 10000000)
        @ops.RegisterGradient(grad_name)
        def _reverse_gradient(op, grad):
            return [tf.negative(grad) * alpha]
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            domain_out = tf.identity(out)
        domain_out = tf.layers.dense(domain_out, 100)
        domain_out = tf.layers.batch_normalization(domain_out, training=is_training)
        domain_out = tf.nn.relu(domain_out)
        domain_logits = tf.layers.dense(domain_out, 2)

    return class_logits, domain_logits


def main(_):
    """Main function for Domain Adaptation by Neural Networks - DANN"""
    tf.reset_default_graph()
    # Load source and target data set
    if FLAGS.source == 'mnist':
        (x_train_s, y_train_s), (x_test_s, y_test_s) = dp.load_mnist(FLAGS.channel_size,
                                                                     FLAGS.truncate_mnist.lower() == 'true')
    elif FLAGS.source == 'mnistm':
        (x_train_s, y_train_s), (x_test_s, y_test_s) = dp.load_mnistm(FLAGS.channel_size)
    elif FLAGS.source == 'svhn':
        (x_train_s, y_train_s), (x_test_s, y_test_s) = dp.load_svhn(FLAGS.channel_size,
                                                                    FLAGS.truncate_svhn.lower() == 'true')
    else:
        sys.exit('For the source set you have to choose one of [svhn, mnist, mnistm]!')
    if FLAGS.target == 'mnist':
        (x_train_t, y_train_t), (x_test_t, y_test_t) = dp.load_mnist(FLAGS.channel_size,
                                                                     FLAGS.truncate_mnist.lower() == 'true')
    elif FLAGS.target == 'mnistm':
        (x_train_t, y_train_t), (x_test_t, y_test_t) = dp.load_mnistm(FLAGS.channel_size)
    elif FLAGS.target == 'svhn':
        (x_train_t, y_train_t), (x_test_t, y_test_t) = dp.load_svhn(FLAGS.channel_size,
                                                                    FLAGS.truncate_svhn.lower() == 'true')
    else:
        sys.exit('For the target set you have to choose one of [svhn, mnist, mnistm]!')

    # Configurations first
    iter_ratio = math.ceil((x_train_s.shape[0] / FLAGS.batch_size))
    print(iter_ratio)
    # We are working with transformed MNIST dataset => image shape is 28x28x3
    feature_columns = [tf.feature_column.numeric_column("x_s", shape=(28, 28, FLAGS.channel_size)),
                       tf.feature_column.numeric_column("x_t", shape=(28, 28, FLAGS.channel_size))]

    # Set up the session config
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    config = tf.estimator.RunConfig(
        save_checkpoints_steps=int(iter_ratio),
        log_step_count_steps=int(iter_ratio),
        session_config=session_config
    )

    # Set up the estimator
    classifier = tf.estimator.Estimator(
        model_fn=estimator_model_fn,
        model_dir=FLAGS.model_dir,
        params={
            'feature_columns': feature_columns,
            'iter_ratio': iter_ratio
        },
        config=config
    )
    if FLAGS.mode == 'train':
        # Set up logging in training mode "test_source_acc": "test_source_acc",
        #                      "test_target_acc": "test_target_acc"
        train_hook = tf.train.LoggingTensorHook(
            tensors={"lr": "learning_rate", "loss": "loss", "source_acc": "source_class_acc",
                     "target_acc": "target_class_acc"},
            every_n_iter=int(iter_ratio))
        # Train and evaluate DANN
        train_spec = tf.estimator.TrainSpec(
            input_fn=tf.estimator.inputs.numpy_input_fn({'x_s': x_train_s, 'x_t': x_train_t},
                                                        {'y_s': y_train_s, 'y_t': y_train_t},
                                                        shuffle=True, batch_size=128, num_epochs=FLAGS.total_epochs),
            max_steps=int(iter_ratio*FLAGS.total_epochs),
            hooks=[train_hook]
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=tf.estimator.inputs.numpy_input_fn({'x_s': x_test_s, 'x_t': x_test_t},
                                                        {'y_s': y_test_s, 'y_t': y_test_t},
                                                        shuffle=True, batch_size=128, num_epochs=1,
                                                        ),
            steps=None,
            throttle_secs=1
        )
        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    elif FLAGS.mode == 'eval':
        classifier.evaluate(
            input_fn=tf.estimator.inputs.numpy_input_fn({'x_s': x_test_s, 'x_t': x_test_t},
                                                        {'y_s': y_test_s, 'y_t': y_test_t},
                                                        shuffle=True, batch_size=128, num_epochs=1,
                                                        )
        )
    else:
        assert FLAGS.mode == 'predict', '-mode flag has to be one of "train", "predict".'


if __name__ == '__main__':
    tf.app.run()
