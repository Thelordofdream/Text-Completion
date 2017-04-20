import tensorflow as tf
import model1


def predict(model, data, sess):
    saver = tf.train.Saver()
    batch_xs, batch_ys = data.next_batch()
    batch_xs = batch_xs.reshape((model.batch_size, model.steps, model.inputs))
    saver.restore(sess, "./model/model.ckpt")
    print sess.run( model.accuracy, feed_dict={model.x: batch_xs, model.y: batch_ys, model.keep_prob: 1.0})
    print(sess.run(tf.argmax(model.output, 1), feed_dict={model.x: batch_xs, model.keep_prob: 1.0}))
    print(sess.run(tf.argmax(batch_ys, 1)))

if __name__ == "__main__":
    training_iters = 3200
    display_step = 10

    data = model1.data(path="./data for input0/")
    my_network = model1.Bd_LSTM_layer(name="TC")

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        predict(my_network, data, sess)
