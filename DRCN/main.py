import tensorflow as tf
import data_pipeline as dp
import numpy as np
import sys
from Model import Model
sys.path.insert(0, '../utilities')
import utilities
from tqdm import tqdm
import os
import pickle

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 128, 'Batch size. ')
tf.flags.DEFINE_integer('total_epochs', 1000, 'Number of epochs for training')
tf.flags.DEFINE_float('learning_rate', 0.0001, 'Constant learning rate')
tf.flags.DEFINE_string('mode', 'train', 'Run the model in "train", "eval" or "predict" mode')
tf.flags.DEFINE_integer('channel_size', 1, 'Either 1 or 3. Defaults to 1.')
tf.flags.DEFINE_float('lambda_', 0.7, 'Lambda trade-off as described in the paper')
tf.flags.DEFINE_string('opt', 'adam', 'Either adam or rmsprop')
tf.flags.DEFINE_string('source', 'mnist', 'Either mnist, mnistm or svhn.')
tf.flags.DEFINE_string('target', 'mnistm', 'Either mnist, mnistm or svhn.')
tf.flags.DEFINE_string('model_dir', './model', 'Where to save the model.')
tf.flags.DEFINE_string('source_only', 'False', 'If set to True, will only train on source classification.')

tf.logging.set_verbosity(tf.logging.DEBUG)


def create_dataset(x_s, y_s, x_t, y_t):
    # Create TensorFlow datasets.
    ds_source_train = tf.data.Dataset.from_tensor_slices((x_s, y_s))
    ds_target_train = tf.data.Dataset.from_tensor_slices((x_t, y_t))
    # Batch all datasets and drop remainder
    ds_source_train = ds_source_train.batch(FLAGS.batch_size, drop_remainder=True)
    ds_target_train = ds_target_train.batch(FLAGS.batch_size, drop_remainder=True)
    return ds_source_train.shuffle(60000), ds_target_train.shuffle(60000)


def main(_):
    # TODO: do not pass source label in target mode (it's not needed!)
    """Main function for Deep-Reconstruction Classification Network - DRCN"""
    tf.reset_default_graph()
    # Load source and target data set
    source_size = 32
    target_size = 32
    if FLAGS.source == 'mnist':
        (x_train_s, y_train_s), (x_test_s, y_test_s) = dp.load_mnist(FLAGS.channel_size, False)
    elif FLAGS.source == 'mnistm':
        (x_train_s, y_train_s), (x_test_s, y_test_s) = dp.load_mnistm(FLAGS.channel_size)
    elif FLAGS.source == 'svhn':
        (x_train_s, y_train_s), (x_test_s, y_test_s) = dp.load_svhn(FLAGS.channel_size, False)
    else:
        sys.exit('For the source set you have to choose one of [svhn, mnist, mnistm]!')

    if FLAGS.target == 'mnist':
        (x_train_t, y_train_t), (x_test_t, y_test_t) = dp.load_mnist(FLAGS.channel_size, False)
    elif FLAGS.target == 'mnistm':
        (x_train_t, y_train_t), (x_test_t, y_test_t) = dp.load_mnistm(FLAGS.channel_size)
    elif FLAGS.target == 'svhn':
        (x_train_t, y_train_t), (x_test_t, y_test_t) = dp.load_svhn(FLAGS.channel_size, False)
    else:
        sys.exit('For the target set you have to choose one of [svhn, mnist, mnistm]!')

    # Create data placeholders.
    placeholder_x_s = tf.placeholder(tf.float32, shape=[None, source_size, source_size, FLAGS.channel_size])
    placeholder_y_s = tf.placeholder(tf.int32, shape=[None])
    placeholder_x_t = tf.placeholder(tf.float32, shape=[None, target_size, target_size, FLAGS.channel_size])
    placeholder_y_t = tf.placeholder(tf.int32, shape=[None])
    placeholder_training = tf.placeholder_with_default(tf.constant(True), shape=[])
    ds_source, ds_target = create_dataset(placeholder_x_s, placeholder_y_s, placeholder_x_t, placeholder_y_t)

    iterator = tf.data.Iterator.from_structure(ds_source.output_types, ds_source.output_shapes)
    x, y = iterator.get_next()

    # Init model
    drcn = Model(FLAGS.opt)
    drcn.train_source(x, y, placeholder_training)
    if FLAGS.source_only.lower() == 'false':
        drcn.train_target(x, y)

    source_iterator = iterator.make_initializer(ds_source)
    target_iterator = iterator.make_initializer(ds_target)

    # Configs
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Stats
    source_acc_train = []
    source_acc_test = []
    target_acc_train = []
    target_acc_test = []
    source_loss_train = []
    target_loss_train = []

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if tf.train.latest_checkpoint(FLAGS.model_dir) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))
        for epoch in range(FLAGS.total_epochs):
            print('Epoch: ', epoch)
            if FLAGS.source_only.lower() == 'false':
                sess.run(target_iterator, feed_dict={placeholder_x_t: x_train_t, placeholder_y_t: y_train_t})
                i = 0
                total_loss = 0
                total_acc = 0
                try:
                    with tqdm(total=len(x_train_t)) as pbar:
                        while True:
                            _, out_loss, source_acc = sess.run([drcn.optimize_reconstruction, drcn.rec_loss,
                                                               drcn.target_class_acc],
                                                               feed_dict={placeholder_training: False})
                            i += 1
                            total_loss += out_loss
                            total_acc += source_acc
                            pbar.update(FLAGS.batch_size)
                            # pbar.write(str(source_acc))
                except tf.errors.OutOfRangeError:
                    print('Done with train target epoch.')
                print(total_acc / i)
                print(total_loss / i)
                target_acc_train.append((epoch, total_acc/float(i)))
                target_loss_train.append((epoch, total_loss/float(i)))
            sess.run(target_iterator, feed_dict={placeholder_x_t: x_test_t, placeholder_y_t: y_test_t})
            i = 0
            total_loss = 0
            total_acc = 0
            try:
                with tqdm(total=len(x_test_t)) as pbar:
                    while True:
                        out_loss, source_acc = sess.run([drcn.source_class_loss, drcn.source_class_acc],
                                                        feed_dict={placeholder_training: False})
                        i += 1
                        total_loss += out_loss
                        total_acc += source_acc
                        pbar.update(FLAGS.batch_size)
            except tf.errors.OutOfRangeError:
                print('Done with evaluation target epoch.')
            print(total_acc / i)
            print(total_loss / i)
            target_acc_test.append((epoch, total_acc / float(i)))
            sess.run(source_iterator, feed_dict={placeholder_x_s: x_train_s, placeholder_y_s: y_train_s})
            i = 0
            total_loss = 0
            total_acc = 0
            try:
                with tqdm(total=len(x_train_s)) as pbar:
                    while True:
                        _, out_loss, source_acc = sess.run([drcn.optimize_class, drcn.source_class_loss,
                                                            drcn.source_class_acc])
                        i += 1
                        total_loss += out_loss
                        total_acc += source_acc
                        pbar.update(FLAGS.batch_size)
            except tf.errors.OutOfRangeError:
                print('Done with source train epoch.')
            print(total_acc/i)
            print(total_loss/i)
            source_acc_train.append((epoch, total_acc/float(i)))
            source_loss_train.append((epoch, total_loss/float(i)))
            sess.run(source_iterator, feed_dict={placeholder_x_s: x_test_s, placeholder_y_s: y_test_s})
            i = 0
            total_loss = 0
            total_acc = 0
            try:
                with tqdm(total=len(x_test_s)) as pbar:
                    while True:
                        out_loss, source_acc = sess.run([drcn.source_class_loss, drcn.source_class_acc],
                                                        feed_dict={placeholder_training: False})
                        i += 1
                        total_loss += out_loss
                        total_acc += source_acc
                        pbar.update(FLAGS.batch_size)
            except tf.errors.OutOfRangeError:
                print('Done with evaluation source epoch.')
            print(total_acc / i)
            print(total_loss / i)
            source_acc_test.append((epoch, total_acc/float(i)))

            saver.save(sess, FLAGS.model_dir, global_step=epoch)
    # Save stats for visualization
    with open(os.path.join(FLAGS.model_dir, 'stats.pkl'), 'wb') as f:
        pickle.dump({'source_acc_train': source_acc_train, 'source_acc_test': source_acc_test,
                     'target_acc_train': target_acc_train, 'target_acc_test': target_acc_test,
                     'source_loss_train': source_loss_train, 'target_loss_train': target_loss_train}, f)


if __name__ == '__main__':
    tf.app.run()
