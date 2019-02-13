import tensorflow as tf
import data_pipeline as dp
FLAGS = tf.flags.FLAGS


class Model(object):

    def __init__(self, optimizer):
        self.source_class_acc = None
        self.source_class_loss = 0
        self.target_class_acc = None
        self.target_class_loss = 0
        self.optimize_class = None
        self.rec_loss = None
        self.optimize_reconstruction = None
        if optimizer == 'adam':
            self.optimizer_class = tf.train.AdamOptimizer(FLAGS.learning_rate, name='adam_class')
            self.optimizer_rec = tf.train.AdamOptimizer(FLAGS.learning_rate, name='adam_reconstruction')
        else:
            self.optimizer_class = tf.train.RMSPropOptimizer(FLAGS.learning_rate, name='rmsprop_class')
            self.optimizer_rec = tf.train.RMSPropOptimizer(FLAGS.learning_rate, name='rmsprop_reconstruction')

    def train_source(self, _input, labels, training):
        # First augment the data
        #_input = tf.cond(training, lambda: dp.augment_data(_input, max_degree=20, max_dx=6, max_dy=6, size=128),
        #                           lambda: _input)
        #if training:
        #    _input = dp.augment_data(_input, max_degree=20, max_dx=6, max_dy=6, size=128)
        # Here we need to call the encoder, classifier and then optimize both according
        # to the respective gradients
        encoded, height, width, channels = self.encoder(_input, is_training=training)
        logits = self.classifier(encoded, is_training=training)

        # Calculate loss and accuracy
        loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
        class_pred = tf.argmax(logits, axis=1, output_type=tf.int32)
        self.source_class_acc = tf.reduce_mean(tf.cast(tf.equal(class_pred, labels), tf.float32))
        # Get variable lists
        var_encoder = tf.trainable_variables('encoder')
        var_classifier = tf.trainable_variables('classifier')

        # Only update the encoder and classifier vars
        self.source_class_loss = loss * FLAGS.lambda_
        tf.summary.scalar("class_loss", self.source_class_loss)
        tf.summary.scalar("source_class_acc", self.source_class_acc)
        tf.summary.scalar("learning_rate", FLAGS.learning_rate)
        tf.summary.merge_all()
        # Perform minimization
        self.optimize_class = self.optimizer_class.minimize(self.source_class_loss, var_list=[var_encoder, var_classifier])

    def train_target(self, _input, labels):
        # First augment the data
        #_input = dp.augment_data(_input, max_degree=20, max_dx=6, max_dy=6, size=128)
        _input_noise = dp.add_noise(_input, noise='zero', level=0.5)
        # Call encoder, classifier and decoder but only train encoder and decoder
        encoded_noise, height, width, channels = self.encoder(_input_noise, is_training=False)
        encoded, _, _, _ = self.encoder(_input, is_training=False)
        # For accuracy reasons only.
        class_logits = self.classifier(encoded, is_training=False)
        class_pred = tf.argmax(class_logits, axis=1, output_type=tf.int32)
        self.target_class_acc = tf.reduce_mean(tf.cast(tf.equal(class_pred, labels), tf.float32))
        tf.summary.scalar("Target_class_acc", self.target_class_acc)
        decoded = self.decoder(encoded_noise, height, width, channels)
        self.rec_loss = tf.losses.mean_squared_error(labels=_input, predictions=decoded) * (1 - FLAGS.lambda_)
        tf.summary.scalar("reconstruction_loss", self.rec_loss)
        tf.summary.merge_all()
        # Get the trainable variables
        var_encoder = tf.trainable_variables('encoder')
        var_decoder = tf.trainable_variables('decoder')
        self.optimize_reconstruction = self.optimizer_rec.minimize(self.rec_loss, var_list=[var_encoder, var_decoder], global_step=tf.train.get_global_step())

    def encoder(self, inputs, is_training=False):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            # inputs = tf.pad(inputs, paddings=tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]))
            net = tf.layers.conv2d(inputs, filters=100, kernel_size=(3, 3), padding='same')
            net = tf.nn.relu(net)
            net = tf.layers.max_pooling2d(net, 2, 2, padding='same')
            # net = tf.pad(net, paddings=tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]))
            net = tf.layers.conv2d(net, filters=150, kernel_size=(3, 3), padding='same')
            net = tf.nn.relu(net)
            net = tf.layers.max_pooling2d(net, 2, 2, padding='same')
            # net = tf.pad(net, paddings=tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]))
            net = tf.layers.conv2d(net, filters=200, kernel_size=(3, 3), padding='same')
            net = tf.nn.relu(net)
            [_, height, width, channels] = net.get_shape().as_list()
            # net = tf.Print(net, [tf.shape(net)], 'Net shape should be ')
            # self.net_shape = tf.print(tf.shape(net))
            net = tf.layers.flatten(net)
            net = tf.layers.dense(net, 1024)
            net = tf.nn.relu(net)
            net = tf.layers.dropout(net, training=is_training)
            net = tf.layers.dense(net, 1024)
            net = tf.nn.relu(net)
        return net, height, width, channels

    def classifier(self, inputs, is_training=False):
        with tf.variable_scope('classifier', reuse=tf.AUTO_REUSE):
            net = tf.layers.dropout(inputs, training=is_training)
            net = tf.layers.dense(net, 10)
        return net

    def decoder(self, inputs, height, width, channels):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(inputs, 1024)
            net = tf.nn.relu(net)
            net = tf.layers.dense(net, height * width * channels)
            net = tf.nn.relu(net)
            # net = tf.Print(net, [tf.shape(net)])
            net = tf.reshape(net, [FLAGS.batch_size, height, width, channels])
            #net = tf.Print(net, [tf.shape(net)])
            # net = tf.pad(net, paddings=tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]))
            net = tf.layers.conv2d(net, filters=200, kernel_size=(3, 3), padding='same')
            net = tf.nn.relu(net)
            # net = tf.pad(net, paddings=tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]))
            net = tf.layers.conv2d(net, filters=150, kernel_size=(5, 5), padding='same')
            net = tf.nn.relu(net)
            net = tf.keras.layers.UpSampling2D(size=(2, 2))(net)
            # net = tf.pad(net, paddings=tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]))
            net = tf.layers.conv2d(net, filters=100, kernel_size=(3, 3), padding='same')
            net = tf.nn.relu(net)
            net = tf.keras.layers.UpSampling2D(size=(2, 2))(net)
            net = tf.layers.conv2d(net, filters=FLAGS.channel_size, kernel_size=(3, 3), padding='same')
            net = tf.clip_by_value(net, 0, 1)
        return net
