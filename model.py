import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


class nerual_network(object):
    def __init__(self, steps=49, inputs=300, hidden=300, batch_size=256, classes=2, learning_rate=0.001):
        self.steps = steps
        self.inputs = inputs
        self.hidden = hidden
        self.batch_size = batch_size
        self.classes = classes
        self.learning_rate = learning_rate


class Bd_LSTM_layer(nerual_network):
    def __init__(self, name="N1"):
        super(Bd_LSTM_layer, self).__init__()
        self.name = name
        self.output = None
        self.cross_entropy = None
        self.optimizer = None
        self.accuracy = None
        self.x = tf.placeholder("float", [None, self.steps, self.inputs], name="x")
        with tf.variable_scope("input_layer"):
            input = self.shape_tranform()

        with tf.variable_scope("Bd_LSTM_layer"):
            _seq_len = tf.fill([self.batch_size], tf.constant(self.steps, dtype=tf.float32))
            with tf.variable_scope("Forward_LSTM"):
                with tf.device("/cpu:0"):
                    lstm_fw_cell = rnn.BasicLSTMCell(self.hidden, forget_bias=0.1, state_is_tuple=True)
                    lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)
            with tf.variable_scope("Backward_LSTM"):
                with tf.device("/cpu:1"):
                    lstm_bw_cell = rnn.BasicLSTMCell(self.hidden, forget_bias=0., state_is_tuple=True)
                    lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)
            outputs, output1, output2 = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input,
                                                                     initial_state_fw=lstm_fw_cell.zero_state(
                                                                         self.batch_size, tf.float32),
                                                                     initial_state_bw=lstm_bw_cell.zero_state(
                                                                         self.batch_size, tf.float32),
                                                                     sequence_length=_seq_len)

        with tf.variable_scope("splice_layer"):
            time_seq = tf.concat([i for i in outputs], 1)

        with tf.variable_scope("hidden_layer1"):
            hidden1_w = tf.Variable(tf.random_normal([self.steps * 2 * self.hidden, self.hidden]), name='h1_w')
            hidden1_b = tf.Variable(tf.random_normal([self.hidden]), name='h1_b'),
            h1 = tf.matmul(time_seq, hidden1_w) + hidden1_b

        with tf.variable_scope("hidden_layer2"):
            hidden2_w = tf.Variable(tf.random_normal([self.hidden, self.classes]), name='h2_w')
            hidden2_b = tf.Variable(tf.random_normal([self.classes]), name='h2_b')
            h2 = tf.matmul(h1, hidden2_w) + hidden2_b

        with tf.variable_scope("dropout"):
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            self.output = tf.nn.dropout(h2, self.keep_prob)

        with tf.variable_scope("loss"):
            self.y = tf.placeholder("float", [None, self.classes], name="y")
            self.cross_entropy= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.output))
        tf.summary.scalar('cross_entropy', self.cross_entropy)

        with tf.variable_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)
            # Evaluate model
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
        _x_train = self.grabVecs(path + "dataset.pkl")
        _y_train = self.grabVecs(path + "label.pkl")
        self.total = len(_x_train)
        list = []
        for i in range(self.total):
            list.append(i)
        np.random.shuffle(list)
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        train = list[:self.total - self.batch_size]
        test = list[self.total - self.batch_size:]
        for i in test:
            self.x_test.append(_x_train[i])
            self.y_test.append(_y_train[i])

        for i in range(4):
            np.random.shuffle(train)
            for i in train:
                self.x_train.append(_x_train[i])
                self.y_train.append(_y_train[i])
        self.total *= 4
        self.start = 0

    def next_batch(self):
        if (self.start + 1) * self.batch_size >= self.total:
            self.start = 0
        batch_x = np.array(self.x_train[self.start * self.batch_size: (self.start + 1) * self.batch_size])
        batch_y = np.array(self.y_train[self.start * self.batch_size: (self.start + 1) * self.batch_size])
        self.start += 1
        return batch_x, batch_y

    def test_batch(self):
        batch_x = np.array(self.x_test)
        batch_y = np.array(self.y_test)
        return batch_x, batch_y

    def grabVecs(self, filename):
        import pickle
        fr = open(filename)
        return pickle.load(fr)
