# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

import tensorflow as tf
from tensorflow import constant
from tensorflow.contrib import rnn
import numpy as np


def grabVecs(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


x_train = grabVecs('./data for input/dataset.pkl')
y_train = grabVecs('./data for input/label.pkl')

# Parameters
learning_rate = 0.001
training_iters = 40
batch_size = 256
display_step = 10

# Network Parameters
n_input = 300  # MNIST data input (img shape: 28*28)
n_steps = 49  # timesteps
n_hidden = 300  # hidden layer num of features
n_classes = 2  # MNIST total classes (0-9 digits)
total = len(x_train)

list = []
for i in range(total):
    list.append(i)
np.random.shuffle(list)
_x_train = []
_y_train = []
_x_test = []
_y_test = []
train = list[:total - batch_size]
test = list[total - batch_size:]
for i in train:
    _x_train.append(x_train[i])
    _y_train.append(y_train[i])
for i in test:
    _x_test.append(x_train[i])
    _y_test.append(y_train[i])

np.random.shuffle(train)
for i in list:
    _x_train.append(x_train[i])
    _y_train.append(y_train[i])

np.random.shuffle(train)
for i in list:
    _x_train.append(x_train[i])
    _y_train.append(y_train[i])

np.random.shuffle(train)
for i in list:
    _x_train.append(x_train[i])
    _y_train.append(y_train[i])

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
istate_fw = tf.placeholder("float", [None, 2 * n_hidden])
istate_bw = tf.placeholder("float", [None, 2 * n_hidden])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'hidden': tf.Variable(tf.random_normal([n_input, 2 * n_hidden]), name='hidden_w'),
    'fc1': tf.Variable(tf.random_normal([n_steps * 2 * n_hidden, n_hidden]), name='fc1_w'),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]), name='out_w'),
}

biases = {
    'hidden': tf.Variable(tf.random_normal([2 * n_hidden]), name='hidden_b'),
    'fc1': tf.Variable(tf.random_normal([n_hidden]), name='fc1_b'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='out_b')
}


def BiRNN(_X, _weights, _biases, _batch_size, _seq_len):
    # BiRNN requires to supply sequence_length as [batch_size, int64]
    # Note: Tensorflow 0.6.0 requires BiRNN sequence_length parameter to be set
    # For a better implementation with latest version of tensorflow, check below
    _seq_len = tf.fill([_batch_size], constant(_seq_len, dtype=tf.float32))
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input])  # (n_steps*batch_size, n_input)
    # Linear activation
    # _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define lstm cells with tensorflow
    # Forward direction cell
    with tf.device("/cpu:0"):
        lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=0.1, state_is_tuple=True)
        lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)
    # Backward direction cell
    with tf.device("/cpu:1"):
        lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=0., state_is_tuple=True)
        lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0)  # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs, output1, output2 = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, _X,
                                                             initial_state_fw=lstm_fw_cell.zero_state(batch_size,
                                                                                                      tf.float32),
                                                             initial_state_bw=lstm_bw_cell.zero_state(batch_size,
                                                                                                      tf.float32),
                                                             sequence_length=_seq_len)
    out1 = tf.concat([i for i in outputs], 1)
    o1 = tf.matmul(out1, _weights['fc1']) + _biases['fc1']
    h_drop = tf.nn.dropout(o1, keep_prob)
    return tf.matmul(h_drop, _weights['out']) + _biases['out']


keep_prob = tf.placeholder(tf.float32)
pred = BiRNN(x, weights, biases, batch_size, n_steps)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))  # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

# loss = tf.nn.seq2seq.sequence_loss_by_example(
#             [pred],
#             [tf.reshape(y, [-1])],
#             [tf.ones([batch_size * n_steps], dtype='int32')])
# cost = tf.reduce_sum(loss) / batch_size
#
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#
# # gradients: return A list of sum(dy/dx) for each x in xs.
# max_grad_norm = 400
# tvars = tf.trainable_variables()
# grads = optimizer.compute_gradients(cost, tvars)
# clipped_grads = tf.clip_by_global_norm(grads, max_grad_norm)

# accept: List of (gradient, variable) pairs, so zip() is needed
# optimizer1 = optimizer.apply_gradients(zip(grads, tvars))


# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session() as sess:
    sess.run(init)
    for i in range(training_iters):
        # 持续迭代
        step = 1
        while step * batch_size <= (total - batch_size) * 4:
            batch_xs = np.array(_x_train[(step - 1) * batch_size: step * batch_size])
            batch_ys = np.array(_y_train[(step - 1) * batch_size: step * batch_size])
            # Reshape data to get 28 seq of 28 elements
            batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                           istate_fw: np.zeros((batch_size, 2 * n_hidden)),
                                           istate_bw: np.zeros((batch_size, 2 * n_hidden)), keep_prob: 0.5})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                    istate_fw: np.zeros((batch_size, 2 * n_hidden)),
                                                    istate_bw: np.zeros((batch_size, 2 * n_hidden)), keep_prob: 1.0})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                                 istate_fw: np.zeros((batch_size, 2 * n_hidden)),
                                                 istate_bw: np.zeros((batch_size, 2 * n_hidden)), keep_prob: 1.0})
                print("Iter " + str(i + 1) + ", Step " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(
                    loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            step += 1
        test_len = batch_size
        test_data = _x_test
        test_label = _y_test
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                                                 istate_fw: np.zeros((test_len, 2 * n_hidden)),
                                                                 istate_bw: np.zeros((test_len, 2 * n_hidden)),
                                                                 keep_prob: 1.0}))
    print("Optimization Finished!")
    # Calculate accuracy for 128 mnist test images
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model/model.ckpt")
    print("Model saved in file: %s" % save_path)
