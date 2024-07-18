import numpy as np
import tensorflow as tf
#from tensorflow.keras import layers, initializers

keras = tf.keras
layers = tf.keras.layers
initializers = tf.keras.initializers


class MAML(tf.Module):
    def __init__(self, d, c, nway, meta_lr=1e-3, train_lr=1e-2):
        super(MAML, self).__init__()
        self.d = d
        self.c = c
        self.nway = nway
        self.meta_lr = meta_lr
        self.train_lr = train_lr
        self.weights = self.conv_weights()
        self.optimizer = tf.keras.optimizers.Adam(self.meta_lr)

        print('img shape:', self.d, self.d, self.c, 'meta-lr:', meta_lr, 'train-lr:', train_lr)

    def build(self, support_xb, support_yb, query_xb, query_yb, K, meta_batchsz, mode='train'):
        self.K = K
        self.meta_batchsz = meta_batchsz
        self.support_x = support_xb
        self.support_y = support_yb
        self.query_x = query_xb
        self.query_y = query_yb
        training = True if mode == 'train' else False

    def save_weights(self, filepath):
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.write(filepath)

    def load_weights(self, filepath):
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.restore(filepath)

    def conv_weights(self):
        weights = {}
        conv_initializer = tf.keras.initializers.GlorotUniform()
        fc_initializer = tf.keras.initializers.GlorotUniform()
        k = 3

        weights['conv1'] = tf.Variable(conv_initializer(shape=[k, k, 3, 32]), name='conv1w')
        weights['b1'] = tf.Variable(tf.zeros([32]), name='conv1b')
        weights['conv2'] = tf.Variable(conv_initializer(shape=[k, k, 32, 32]), name='conv2w')
        weights['b2'] = tf.Variable(tf.zeros([32]), name='conv2b')
        weights['conv3'] = tf.Variable(conv_initializer(shape=[k, k, 32, 32]), name='conv3w')
        weights['b3'] = tf.Variable(tf.zeros([32]), name='conv3b')
        weights['conv4'] = tf.Variable(conv_initializer(shape=[k, k, 32, 32]), name='conv4w')
        weights['b4'] = tf.Variable(tf.zeros([32]), name='conv4b')

        weights['w5'] = tf.Variable(fc_initializer(shape=[32 * 5 * 5, self.nway]), name='fc1w')
        weights['b5'] = tf.Variable(tf.zeros([self.nway]), name='fc1b')

        return weights

    def conv_block(self, x, weight, bias, scope, training):
        x = tf.nn.conv2d(x, weight, [1, 1, 1, 1], 'SAME') + bias
        x = layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        return x

    def forward(self, x, weights, training):
        x = tf.reshape(x, [-1, self.d, self.d, self.c])

        hidden1 = self.conv_block(x, weights['conv1'], weights['b1'], 'conv0', training)
        hidden2 = self.conv_block(hidden1, weights['conv2'], weights['b2'], 'conv1', training)
        hidden3 = self.conv_block(hidden2, weights['conv3'], weights['b3'], 'conv2', training)
        hidden4 = self.conv_block(hidden3, weights['conv4'], weights['b4'], 'conv3', training)

        hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
        output = tf.add(tf.matmul(hidden4, weights['w5']), weights['b5'])

        return output
