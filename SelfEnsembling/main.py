import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import data_pipeline as dp
import math
import sys
sys.path.insert(0, '../utilities')
import utilities
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 128, 'Batch size.')
tf.flags.DEFINE_integer('total_epochs', 100, 'Number of epochs for training.')
tf.flags.DEFINE_float('learning_rate', 0.01, 'Base learning rate for learning rate decay.')
tf.flags.DEFINE_integer('rampup_epochs', 80, 'Number of epochs for ramp up factor.')

tf.flags.DEFINE_float('self_ensembling_loss_weight', 3.0, 'Self ensembling loss weight.')
tf.flags.DEFINE_float('confidence_threshold', 0.968, 'Confidence threshold.')
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
    training = mode == tf.estimator.ModeKeys.TRAIN

    input_layer_source = tf.feature_column.input_layer({"x_s": features['x_s']}, params['feature_columns'][0])
    input_layer_target = tf.feature_column.input_layer({"x_t": features['x_t']}, params['feature_columns'][1])

    # Reshape
    input_layer_source = tf.reshape(input_layer_source, [-1, params['source_size'], params['source_size'],
                                                         FLAGS.channel_size])
    input_layer_target = tf.reshape(input_layer_target, [-1, params['target_size'], params['target_size'],
                                                         FLAGS.channel_size])
    # Apply random horizontal flipping and random crops after zero padding

    y_s = tf.cast(labels['y_s'], tf.int32)
    y_t = tf.cast(labels['y_t'], tf.int32)


    if mode == tf.estimator.ModeKeys.PREDICT:
        # TODO: To be implemented
        return
    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        # Prepare the three different pipelines
        if training:
            input_layer_source_aug = augment_input(input_layer_source, params['source_size'])
            input_layer_target_aug_student = augment_input(input_layer_target, params['source_size'])
            input_layer_target_aug_teacher = augment_input(input_layer_target, params['source_size'])
            with tf.control_dependencies([input_layer_source_aug]):
                start = tf.timestamp()
        else:
            input_layer_source_aug = input_layer_source
            input_layer_target_aug_student = input_layer_target
            input_layer_target_aug_teacher = input_layer_target

        # Initialize the exponential moving average
        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        # Apply self ensembling for the student network
        class_logits_source_student = self_ensembling_fn(input_layer_source_aug, scope='classifier',
                                                         is_training=training)
        # class_logits_source_teacher = self_ensembling_fn(input_layer_source_aug, scope='classifier',
        #                                                  is_training=training, getter=get_getter(ema))

        var_class = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        class_logits_target_student = self_ensembling_fn(input_layer_target_aug_student, scope='classifier',
                                                         is_training=training)
        class_logits_target_teacher = self_ensembling_fn(input_layer_target_aug_teacher, scope='classifier',
                                                         is_training=training, getter=get_getter(ema))
        # with tf.control_dependencies([class_logits_target_teacher]):
        #     class_logits_target_teacher = tf.Print(class_logits_target_teacher, [start-tf.timestamp()], "Current time: ", summarize=1000)
        # Get predictions for accuracy
        pred_classes_source_student = tf.argmax(class_logits_source_student, axis=1, output_type=tf.int32)
        # pred_classes_source_teacher = tf.argmax(class_logits_source_teacher, axis=1, output_type=tf.int32)
        pred_classes_target_student = tf.argmax(class_logits_target_student, axis=1, output_type=tf.int32)
        pred_classes_target_teacher = tf.argmax(class_logits_target_teacher, axis=1, output_type=tf.int32)

        # Compute losses
        class_loss_source = tf.losses.sparse_softmax_cross_entropy(labels=y_s, logits=class_logits_source_student)
        if training:
            if FLAGS.rampup_epochs > 0:
                squared_difference_loss = tf.losses.mean_squared_error(tf.nn.softmax(class_logits_target_student),
                                                                       tf.nn.softmax(class_logits_target_teacher))
            else:
                max_target_teacher = tf.reduce_max(class_logits_target_teacher, axis=1)
                binary_mask = tf.cast(math_ops.greater(max_target_teacher, FLAGS.confidence_threshold),
                                      tf.float32)
                loss = class_logits_target_student - class_logits_target_teacher
                loss = loss * loss
                loss = tf.reduce_mean(loss, axis=1)
                squared_difference_loss = tf.reduce_mean(loss * binary_mask)
        # Ramp up squared difference loss
        if training:
            if FLAGS.rampup_epochs > 0:
                iter_ratio = params['iter_ratio']
                current_epoch = math_ops.ceil(math_ops.divide(tf.train.get_global_step(), iter_ratio))
                rampup = utilities.calculate_ramp_up(current_epoch, FLAGS.rampup_epochs)
                squared_difference_loss = squared_difference_loss * rampup

            # Compute weighted loss
            total_loss = class_loss_source + squared_difference_loss * FLAGS.self_ensembling_loss_weight
        else:
            total_loss = class_loss_source

        if mode == tf.estimator.ModeKeys.EVAL:
            class_logits_source_teacher = self_ensembling_fn(input_layer_source_aug, scope='classifier',
                                                             is_training=training, getter=get_getter(ema))
            pred_classes_source_teacher = tf.argmax(class_logits_source_teacher, axis=1, output_type=tf.int32)
            source_class_acc = tf.metrics.accuracy(labels=y_s,
                                                   predictions=pred_classes_source_teacher,
                                                   name='source_class_acc_op')
            target_class_acc = tf.metrics.accuracy(labels=y_t,
                                                   predictions=pred_classes_target_teacher,
                                                   name='source_class_acc_op')
            metrics = {'source_class_acc': source_class_acc, 'target_class_acc': target_class_acc}
            return tf.estimator.EstimatorSpec(
                mode, loss=total_loss, eval_metric_ops=metrics)

        # Calculate a non streaming (per batch) accuracy
        source_class_acc_student = utilities.non_streaming_accuracy(pred_classes_source_student, y_s)
        target_class_acc_student = utilities.non_streaming_accuracy(pred_classes_target_student, y_t)
        # source_class_acc_teacher = utilities.non_streaming_accuracy(pred_classes_source_teacher, y_s)
        target_class_acc_teacher = utilities.non_streaming_accuracy(pred_classes_target_teacher, y_t)

        tf.identity(FLAGS.learning_rate, 'learning_rate')
        tf.identity(total_loss, 'loss')
        tf.identity(source_class_acc_student, 'source_class_acc_student')
        tf.identity(target_class_acc_student, 'target_class_acc_student')
        # tf.identity(source_class_acc_teacher, 'source_class_acc_teacher')
        tf.identity(target_class_acc_teacher, 'target_class_acc_teacher')
        # TensorBoard
        tf.summary.scalar('Train_source_acc_student', source_class_acc_student)
        tf.summary.scalar('Train_target_acc_student', target_class_acc_student)
        # tf.summary.scalar('Train_source_acc_teacher', source_class_acc_teacher)
        tf.summary.scalar('Train_target_acc_teacher', target_class_acc_teacher)
        tf.summary.scalar('Learning_rate', FLAGS.learning_rate)
        tf.summary.merge_all()

        # Optimize
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())
        with tf.control_dependencies([train_op]):
            ema_op = ema.apply(var_class)
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=ema_op)


def self_ensembling_fn(input_layer, scope, is_training=False, getter=None):
    if FLAGS.source in ['mnist', 'mnistm'] and FLAGS.target != 'svhn':
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, custom_getter=getter):
            out = tf.layers.conv2d(input_layer, filters=64, kernel_size=5, padding='SAME')
            out = tf.layers.batch_normalization(out, training=is_training)
            out = tf.layers.max_pooling2d(out, 3, 2)
            out = tf.nn.relu(out)
            # out = tf.layers.dropout(out, rate=0.8, training=is_training)
            out = tf.layers.conv2d(out, filters=64, kernel_size=5, padding='SAME')
            out = tf.layers.batch_normalization(out, training=is_training)
            # out = tf.layers.dropout(out, training=is_training)
            out = tf.layers.max_pooling2d(out, 3, 2)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, filters=128, kernel_size=5, padding='SAME')
            out = tf.layers.batch_normalization(out, training=is_training)
            out = tf.nn.relu(out)
            # out = tf.layers.dropout(out, rate=0.5, training=is_training)
            out = tf.layers.flatten(out)

            out = tf.layers.dense(out, 3072)
            out = tf.layers.batch_normalization(out, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.dropout(out, training=is_training)
            out = tf.layers.dense(out, 2048)
            out = tf.layers.batch_normalization(out, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.dense(out, units=10, name='class_logit')
            # out = tf.Print(out, [tf.shape(out)], "Out shape after fc ", summarize=100)
        return out
    else:
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, custom_getter=getter):
            out = tf.layers.conv2d(input_layer, filters=128, kernel_size=3, padding='SAME')
            out = tf.layers.batch_normalization(out, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, filters=128, kernel_size=3, padding='SAME')
            out = tf.layers.batch_normalization(out, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, filters=128, kernel_size=3, padding='SAME')
            out = tf.layers.batch_normalization(out, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, pool_size=2, strides=2)
            out = tf.layers.dropout(out, rate=0.5, training=is_training)

            out = tf.layers.conv2d(out, filters=256, kernel_size=3, padding='SAME')
            out = tf.layers.batch_normalization(out, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, filters=256, kernel_size=3, padding='SAME')
            out = tf.layers.batch_normalization(out, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, filters=256, kernel_size=3, padding='SAME')
            out = tf.layers.batch_normalization(out, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, pool_size=2, strides=2)
            out = tf.layers.dropout(out, rate=0.5, training=is_training)

            out = tf.layers.conv2d(out, filters=512, kernel_size=3)
            out = tf.layers.batch_normalization(out, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, filters=256, kernel_size=1)
            out = tf.layers.batch_normalization(out, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, filters=128, kernel_size=1)
            out = tf.layers.batch_normalization(out, training=is_training)
            out = tf.nn.relu(out)
            # out = tf.Print(out, [tf.shape(out)], "Out shape before reduce mean ", summarize=100)

            # Easier to use reduce_mean instead of avg_pool2d
            out = tf.reduce_mean(out, axis=[1, 2])
            # out = tf.Print(out, [tf.shape(out)], "Out shape before fc ", summarize=100)
            out = tf.layers.dense(out, 10)
            # out = tf.Print(out, [tf.shape(out)], "Out shape after fc ", summarize=100)
        return out

def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter


def augment_input(inputs, image_size):
    input_layer_source = inputs
    # input_layer_source = tf.image.random_flip_left_right(inputs)
    # input_layer_source = tf.pad(input_layer_source, paddings=tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]))
    # input_layer_source = tf.random_crop(input_layer_source,
    #                                     [tf.shape(input_layer_source)[0], image_size, image_size, FLAGS.channel_size])
    return input_layer_source


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
    feature_columns = [tf.feature_column.numeric_column("x_s", shape=(32, 32, FLAGS.channel_size)),
                       tf.feature_column.numeric_column("x_t", shape=(32, 32, FLAGS.channel_size))]

    # Set up the session config
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    config = tf.estimator.RunConfig(
        save_checkpoints_steps=int(iter_ratio),
        log_step_count_steps=100,
        session_config=session_config
    )

    # Set up the estimator
    classifier = tf.estimator.Estimator(
        model_fn=estimator_model_fn,
        model_dir=FLAGS.model_dir,
        params={
            'feature_columns': feature_columns,
            'iter_ratio': iter_ratio,
            'source_size': 32,
            'target_size': 32
        },
        config=config
    )
    if FLAGS.mode == 'train':
        # Set up logging in training mode "test_source_acc": "test_source_acc",
        #                      "test_target_acc": "test_target_acc"
        train_hook = tf.train.LoggingTensorHook(
            tensors={"lr": "learning_rate", "loss": "loss", "source_class_acc_student": "source_class_acc_student",
                     "target_class_acc_student": "target_class_acc_student",
                     "target_class_acc_teacher": "target_class_acc_teacher"},
            every_n_iter=100)
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
