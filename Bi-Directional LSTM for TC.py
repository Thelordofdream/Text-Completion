# coding=utf-8
import tensorflow as tf
import model


def train(model, data, sess, training_iters, display_step):
    train_writer = tf.summary.FileWriter('./train', sess.graph)
    sess.run(init)
    step = 1
    while step < training_iters:
        batch_xs, batch_ys = data.next_batch()
        batch_xs = batch_xs.reshape((model.batch_size, model.steps, model.inputs))
        sess.run(model.optimizer, feed_dict={model.x: batch_xs, model.y: batch_ys, model.keep_prob: 0.5})
        if step % display_step == 0:
            summary, acc, loss = sess.run([model.merged, model.accuracy, model.cross_entropy], feed_dict={model.x: batch_xs,
                                                                                           model.y: batch_ys,
                                                                                           model.keep_prob: 1.0})
            train_writer.add_summary(summary, step)
            print("Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(
                loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")


def test(model, data, sess):
    test_data, test_label = data.test_batch()
    test_data = test_data.reshape((-1, model.steps, model.inputs))
    print("Testing Accuracy:", sess.run(model.accuracy, feed_dict={model.x: test_data, model.y: test_label, model.keep_prob: 1.0}))


def save(sess):
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./test/model.ckpt")
    print("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    training_iters = 3200
    display_step = 10

    data = model.data(path="./data for input/")
    my_network = model.Bd_LSTM_layer(name="TC")

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        train(my_network, data, sess, training_iters, display_step)
        test(my_network, data, sess)
        save(sess)