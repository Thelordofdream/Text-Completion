import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


class nerual_network(object):
    def __init__(self, steps=49, inputs=300, hidden=300, batch_size=1, classes=2, learning_rate=0.001):
        self.steps = steps
        self.inputs = inputs
        self.hidden = hidden
        self.batch_size = batch_size
        self.classes = classes
        self.learning_rate = learning_rate


class RNN_layer(nerual_network):
    def __init__(self, name="N1"):
        super(RNN_layer, self).__init__()
        self.name = name
        self.output = None
        self.cross_entropy = None
        self.optimizer = None
        self.accuracy = None
        self.x = tf.placeholder("float", [None, self.steps, self.inputs], name="x")
        with tf.variable_scope("input_layer"):
            input = self.shape_tranform()

        with tf.variable_scope("RNN_layer"):
            _seq_len = tf.fill([self.batch_size], tf.constant(self.steps, dtype=tf.float32))
            lstm_cell = rnn.BasicLSTMCell(self.hidden, forget_bias=1.0, state_is_tuple=True)
            outputs, states = rnn.static_rnn(lstm_cell, input, initial_state=lstm_cell.zero_state(self.batch_size, tf.float32))

        with tf.variable_scope("dense_layer"):
            outputs = tf.transpose(outputs, [1, 0, 2])
            time_seq = tf.reshape(outputs, [-1, self.steps * 2 * self.inputs])
            hidden1_w = tf.Variable(tf.random_normal([self.steps * 2 * self.hidden, self.hidden]), name='h1_w')
            hidden1_b = tf.Variable(tf.random_normal([self.hidden]), name='h1_b'),
            h1 = tf.matmul(time_seq, hidden1_w) + hidden1_b

        with tf.variable_scope("dropout"):
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            h_drop = tf.nn.dropout(h1, self.keep_prob)

        with tf.variable_scope("readout_layer"):
            hidden2_w = tf.Variable(tf.random_normal([self.hidden, self.classes]), name='h2_w')
            hidden2_b = tf.Variable(tf.random_normal([self.classes]), name='h2_b')
            self.output = tf.matmul(h_drop, hidden2_w) + hidden2_b

        self.y = tf.placeholder("float", [None, self.classes], name="y")
        with tf.variable_scope("loss"):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.output))
        tf.summary.scalar('cross_entropy', self.cross_entropy)

        with tf.variable_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)
            correct_pred = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def shape_tranform(self):
        X = tf.transpose(self.x, [1, 0, 2])
        X = tf.reshape(X, [-1, self.inputs])
        X = tf.split(X, self.steps, 0)
        return X


class data(nerual_network):
    def __init__(self, path):
        super(data, self).__init__()
        self._x_train = self.grabVecs(path + "dataset.pkl")
        self._y_train = self.grabVecs(path + "label.pkl")
        self.total0 = len(self._x_train)
        self.rest = 4
        random_list = []
        for i in range(self.total0):
            random_list.append(i)
        np.random.shuffle(random_list)
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        train = random_list[:self.total0 - self.batch_size]
        test = random_list[self.total0 - self.batch_size:]
        for i in test:
            self.x_test.append(self._x_train[i])
            self.y_test.append(self._y_train[i])

        for i in range(self.rest):
            np.random.shuffle(train)
            for i in train:
                self.x_train.append(self._x_train[i])
                self.y_train.append(self._y_train[i])

        for i in range(self.rest // 2):
            for i in random_list:
                self.x_train.append(self._x_train[i])
                self.y_train.append(self._y_train[i])

        self.total = len(self.x_train)
        self.start = 0
        self.start0 = 0
        self.max = self.total // self.batch_size
        self.max0 = self.total0 // self.batch_size

    def next_batch(self):
        batch_x = np.array(
            self.x_train[(self.start % self.max) * self.batch_size: (self.start % self.max + 1) * self.batch_size])
        batch_y = np.array(
            self.y_train[(self.start % self.max) * self.batch_size: (self.start % self.max + 1) * self.batch_size])
        self.start += 1
        return batch_x, batch_y

    def test_batch(self):
        batch_x = np.array(self.x_test)
        batch_y = np.array(self.y_test)
        return batch_x, batch_y

    def next_predict_batch(self):
        batch_x = np.array(
            self._x_train[(self.start0 % self.max0) * self.batch_size: (self.start0 % self.max0 + 1) * self.batch_size])
        batch_y = np.array(
            self._y_train[(self.start0 % self.max0) * self.batch_size: (self.start0 % self.max0 + 1) * self.batch_size])
        self.start0 += 1
        return batch_x, batch_y

    def grabVecs(self, filename):
        import pickle
        fr = open(filename)
        return pickle.load(fr)
