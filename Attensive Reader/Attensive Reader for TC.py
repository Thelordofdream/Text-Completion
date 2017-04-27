# coding=utf-8
import model2
import tensorflow as tf


def train(model, data, sess, training_iters, display_step):
    train_writer = tf.summary.FileWriter('./train1', sess.graph)
    sess.run(init)
    step = 0
    while step < training_iters:
        batch_xs, batch_qs, batch_as, batch_ys = data.next_batch()
        batch_xs = batch_xs.reshape((model.batch_size, model.steps, model.inputs))
        batch_qs = batch_qs.reshape((model.batch_size, model.steps, model.inputs))
        batch_as = batch_as.reshape((model.batch_size, 4, model.inputs))
        sess.run(model.optimizer, feed_dict={model.x: batch_xs, model.q: batch_qs, model.a: batch_as, model.y: batch_ys,
                                             model.keep_prob_d: 0.5, model.keep_prob_q: 0.5, model.keep_prob_a: 0.5})
        if step % display_step == 0:
            summary, acc, loss = sess.run([model.merged, model.accuracy, model.cross_entropy],
                                          feed_dict={model.x: batch_xs,
                                                     model.q: batch_qs,
                                                     model.a: batch_as,
                                                     model.y: batch_ys,
                                                     model.keep_prob_d: 1.0,
                                                     model.keep_prob_q: 1.0,
                                                     model.keep_prob_a: 1.0})
            train_writer.add_summary(summary, step)
            print("Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(
                loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            test(model, data, sess)
        step += 1
    print("Optimization Finished!")


def test(model, data, sess):
    test_data, test_q, test_a, test_label = data.test_batch()
    test_data = test_data.reshape((-1, model.steps, model.inputs))
    test_q = test_q.reshape((-1, model.steps, model.inputs))
    test_a = test_a.reshape((-1, 4, model.inputs))
    print("Testing Accuracy:",
          sess.run(model.accuracy,
                   feed_dict={model.x: test_data, model.q: test_q, model.a: test_a, model.y: test_label,
                              model.keep_prob_d: 1.0, model.keep_prob_q: 1.0, model.keep_prob_a: 1.0}))


def save(sess):
    saver = tf.train.Saver()
    save_path = saver.save(sess, "../test1/model.ckpt")
    print("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    training_iters = 10
    display_step = 1

    data = model2.data(path="../data for input0/")
    my_network = model2.Attensive_Reader(name="TC")

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        train(my_network, data, sess, training_iters, display_step)
        test(my_network, data, sess)
        save(sess)
