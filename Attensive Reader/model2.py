import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


class nerual_network(object):
    def __init__(self, steps=49, inputs=300, hidden_d=300, hidden_q=64, batch_size=1, classes=2, learning_rate=0.001):
        self.steps = steps
        self.inputs = inputs
        self.hidden_d = hidden_d
        self.hidden_q = hidden_q
        self.batch_size = batch_size
        self.classes = classes
        self.learning_rate = learning_rate


class Attensive_Reader(nerual_network):
    def __init__(self, name="N1"):
        super(Attensive_Reader, self).__init__()
        self.name = name
        self.output = None
        self.cross_entropy = None
        self.optimizer = None
        self.accuracy = None
        self.x = tf.placeholder("float", [None, self.steps, self.inputs], name="x")
        with tf.variable_scope("input_layer"):
            input_d = self.shape_tranform(self.x, self.steps)

        with tf.variable_scope("Bd_LSTM_layer"):
            outputs_d, output1_d, output2_d = self.create_LSTM_layer(input_d, seq_len=self.steps)

        with tf.variable_scope("dense_layer"):
            outputs_d = tf.transpose(outputs_d, [1, 0, 2])
            time_seq = tf.reshape(outputs_d, [-1, self.steps * 2 * self.inputs])
            hidden1_d_w = tf.Variable(tf.random_normal([self.steps * 2 * self.inputs, self.hidden_d]), name='hd1_w')
            hidden1_d_b = tf.Variable(tf.random_normal([self.hidden_d]), name='hd1_b'),
            hd_1 = tf.matmul(time_seq, hidden1_d_w) + hidden1_d_b

        with tf.variable_scope("dropout"):
            self.keep_prob_d = tf.placeholder(tf.float32, name="keep_prob_d")
            hd_drop = tf.nn.dropout(hd_1, self.keep_prob_d)

        with tf.variable_scope("readout_layer"):
            hidden2_d_w = tf.Variable(tf.random_normal([self.hidden_d, self.classes]), name='hd2_w')
            hidden2_d_b = tf.Variable(tf.random_normal([self.classes]), name='hd2_b')
            output_d = tf.matmul(hd_drop, hidden2_d_w) + hidden2_d_b

        self.q = tf.placeholder("float", [None, self.steps, self.inputs], name="q")
        with tf.variable_scope("input_layer_q"):
            input_q = self.shape_tranform(self.q, self.steps)

        with tf.variable_scope("Q_LSTM_layer"):
            outputs_q, output1_q, output2_q = self.create_LSTM_layer(input_q, seq_len=self.steps)

        with tf.variable_scope("hidden_layer_q"):
            outputs_q = tf.reshape(outputs_q, [-1, 2 * self.inputs])  # (n_steps*batch_size, n_input)
            hidden1_q_w = tf.Variable(tf.random_normal([2 * self.inputs, 2 * self.hidden_q]), name='hq1_w')
            hidden1_q_b = tf.Variable(tf.random_normal([2 * self.hidden_q]), name='hq1_b'),
            hq_1 = tf.matmul(outputs_q, hidden1_q_w) + hidden1_q_b
            hq_1 = tf.split(hq_1, self.steps, 0)

        with tf.variable_scope("dropout_q"):
            self.keep_prob_q = tf.placeholder(tf.float32, name="keep_prob_q")
            hq1_drop = tf.nn.dropout(hq_1, self.keep_prob_q)

        self.a = tf.placeholder("float", [None, 4, self.inputs], name="a")
        with tf.variable_scope("input_layer_a"):
            input_a = self.shape_tranform(self.a, 4)

        with tf.variable_scope("A_LSTM_layer"):
            outputs_a, output1_a, output2_a = self.create_LSTM_layer(input_a, seq_len=4)

        with tf.variable_scope("hidden_layer_a_fw"):
            hidden1_a_w = tf.Variable(tf.random_normal([self.inputs, self.hidden_q]), name='ha1_w')
            hidden1_a_b = tf.Variable(tf.random_normal([self.hidden_q]), name='ha1_b')
            output1_a = tf.reshape(output1_a, [-1, self.inputs])
            ha1_fw = tf.matmul(output1_a, hidden1_a_w) + hidden1_a_b
            ha1_fw = tf.split(ha1_fw, 2, 0)

        with tf.variable_scope("hidden_layer_a_bw"):
            output2_a = tf.reshape(output2_a, [-1, self.inputs])
            ha1_bw = tf.matmul(output2_a, hidden1_a_w) + hidden1_a_b
            ha1_bw = tf.split(ha1_bw, 2, 0)

        ha_1 = tf.concat([ha1_fw[0], ha1_bw[0]], 1, name="concat")

        with tf.variable_scope("dropout_a"):
            self.keep_prob_a = tf.placeholder(tf.float32, name="keep_prob_a")
            ha1_drop = tf.nn.dropout(ha_1, self.keep_prob_d)

        with tf.variable_scope("attention_layer"):
            Wum = tf.Variable(tf.random_normal([2 * self.hidden_q, 2 * self.hidden_q]), name='Wum')
            mu = tf.matmul(ha_1, Wum)
            m = []
            Wym = tf.Variable(tf.random_normal([2 * self.hidden_q, 2 * self.hidden_q]), name='Wym')
            for i in range(self.steps):
                m.append(tf.nn.tanh(tf.matmul(hq1_drop[i], Wym) + mu))
            m = tf.reshape(m, [-1, 2 * self.hidden_q])
            Wms = tf.Variable(tf.random_normal([2 * self.hidden_q, 1]), name='Wms')
            self.s = tf.placeholder("float", [None, self.steps], name="s")
            s0 = tf.matmul(m, Wms)
            s0 = tf.split(s0, self.steps, 0)
            hq1_drop = tf.transpose(hq1_drop, [1, 0, 2])
            s0 = tf.reshape(s0, [self.batch_size, self.steps])
            s0 = tf.nn.softmax(s0)
            self.s = tf.reshape(s0, [self.batch_size, self.steps, 1])
            r = []
            for i in range(self.batch_size):
                r.append(tf.transpose(tf.matmul(tf.transpose(hq1_drop[i]), self.s[i])))
            r = tf.reshape(r, [-1, 2 * self.hidden_q])
            r_drop = tf.nn.dropout(r, self.keep_prob_q)

        with tf.variable_scope("keyword_layer"):
            Wug = tf.Variable(tf.random_normal([2 * self.hidden_q, self.classes]), name='Wug')
            Wrg = tf.Variable(tf.random_normal([2 * self.hidden_q, self.classes]), name='Wrg')
            output_g = tf.nn.tanh(tf.matmul(ha1_drop, Wug) + tf.matmul(r_drop, Wrg))

        self.output = tf.add(output_d, output_g, name="add")

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

    def shape_tranform(self, x, steps):
        X = tf.transpose(x, [1, 0, 2])
        X = tf.reshape(X, [-1, self.inputs])
        X = tf.split(X, steps, 0)
        return X

    def create_LSTM_layer(self, input, seq_len):
        _seq_len = tf.fill([self.batch_size], tf.constant(seq_len, dtype=tf.float32))
        with tf.variable_scope("Forward_LSTM"):
            with tf.device("/cpu:0"):
                lstm_fw_cell = rnn.BasicLSTMCell(self.inputs, forget_bias=0.1, state_is_tuple=True)
                lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)
        with tf.variable_scope("Backward_LSTM"):
            with tf.device("/cpu:1"):
                lstm_bw_cell = rnn.BasicLSTMCell(self.inputs, forget_bias=0., state_is_tuple=True)
                lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)
        outputs, output1, output2 = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input,
                                                                 initial_state_fw=lstm_fw_cell.zero_state(
                                                                     self.batch_size, tf.float32),
                                                                 initial_state_bw=lstm_bw_cell.zero_state(
                                                                     self.batch_size, tf.float32),
                                                                 sequence_length=_seq_len)
        return outputs, output1, output2


class data(nerual_network):
    def __init__(self, path):
        super(data, self).__init__()
        _x_train = self.grabVecs(path + "dataset.pkl")
        _q_train = self.grabVecs(path + "q.pkl")
        _a_train = self.grabVecs(path + "a.pkl")
        _y_train = self.grabVecs(path + "label.pkl")
        self.total = len(_x_train)
        self.rest = 4
        random_list = []
        for i in range(self.total):
            random_list.append(i)
        np.random.shuffle(random_list)
        self.x_train = []
        self.q_train = []
        self.a_train = []
        self.y_train = []
        self.x_test = []
        self.q_test = []
        self.a_test = []
        self.y_test = []
        train = random_list[:self.total - self.batch_size]
        test = random_list[self.total - self.batch_size:]
        for i in test:
            self.x_test.append(_x_train[i])
            self.q_test.append(_q_train[i])
            self.a_test.append(_a_train[i])
            self.y_test.append(_y_train[i])

        for i in range(self.rest):
            np.random.shuffle(train)
            for i in train:
                self.x_train.append(_x_train[i])
                self.q_train.append(_q_train[i])
                self.a_train.append(_a_train[i])
                self.y_train.append(_y_train[i])
        self.total = len(self.x_train)
        self.start = 0
        self.max = self.total // self.batch_size

    def next_batch(self):
        batch_x = np.array(
            self.x_train[(self.start % self.max) * self.batch_size: (self.start % self.max + 1) * self.batch_size])
        batch_q = np.array(
            self.q_train[(self.start % self.max) * self.batch_size: (self.start % self.max + 1) * self.batch_size])
        batch_a = np.array(
            self.a_train[(self.start % self.max) * self.batch_size: (self.start % self.max + 1) * self.batch_size])
        batch_y = np.array(
            self.y_train[(self.start % self.max) * self.batch_size: (self.start % self.max + 1) * self.batch_size])
        self.start += 1
        return batch_x, batch_q, batch_a, batch_y

    def test_batch(self):
        batch_x = np.array(self.x_test)
        batch_q = np.array(self.q_test)
        batch_a = np.array(self.a_test)
        batch_y = np.array(self.y_test)
        return batch_x, batch_q, batch_a, batch_y

    def grabVecs(self, filename):
        import pickle
        fr = open(filename)
        return pickle.load(fr)
