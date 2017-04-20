import tensorflow as tf
import model1
import gensim
import Word2Vec
import numpy as np


def predict(model, data, sess):
    saver = tf.train.Saver()
    results = []
    for batch_xs in data:
        batch_xs = batch_xs.reshape((model.batch_size, model.steps, model.inputs))
        saver.restore(sess, "./model/model.ckpt")
        results.append(sess.run(model.output, feed_dict={model.x: batch_xs, model.keep_prob: 1.0}))
    return results


def generate(q1, q2, answer, model_google):
    sentences = []
    for i in answer:
        sentences.append(q1 + answer[i] + q2)
    sentences = Word2Vec.cleanText(sentences)
    print sentences
    n_dim = 300
    vectors = [Word2Vec.buildWordVector(model_google, z, n_dim) for z in sentences]
    print len(vectors[0])
    dataset = []
    for a in vectors:
        sentence = np.zeros((49, 300))
        m = len(a)
        start = int((49 - m) / 2)
        sentence[start:start + m] = a
        dataset.append(np.array(sentence))
    return dataset


if __name__ == "__main__":
    model_google = gensim.models.KeyedVectors.load_word2vec_format('./GoogleModel/GoogleNews-vectors-negative300.bin', binary=True)
    question1 = "It is a paradox of the Victorians that they were both"
    question2 = "and, through their empire, cosmopolitan."
    answer = {}
    answer["A"] = "capricious"
    answer["B"] = "insular"
    answer["C"] = "mercenary"
    answer["D"] = "idealistic"
    answer["E"] = "intransigent"

    data = generate(question1, question2, answer, model_google)
    my_network = model1.Bd_LSTM_layer(name="TC")
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        results = predict(my_network, data, sess)
    print results
