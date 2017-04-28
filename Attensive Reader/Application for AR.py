import time

import gensim
import numpy as np
import pymysql
import tensorflow as tf

import Word2Vec
import model2
from draw import draw


def predict(model, data, q, a, sess):
    results = []
    points = []
    for num in range(len(data)):
        batch_xs = data[num].reshape((model.batch_size, model.steps, model.inputs))
        batch_q = q[num].reshape((model.batch_size, model.steps, model.inputs))
        batch_a = a[num].reshape((model.batch_size, 4, model.inputs))
        results.append(sess.run(tf.argmax(model.output, 1), feed_dict={model.x: batch_xs, model.q: batch_q, model.a: batch_a, model.keep_prob_d: 1.0,  model.keep_prob_q: 1.0,  model.keep_prob_a: 1.0}))
        points.append((sess.run(model.output, feed_dict={model.x: batch_xs, model.q: batch_q, model.a: batch_a, model.keep_prob_d: 1.0,  model.keep_prob_q: 1.0,  model.keep_prob_a: 1.0})))
    return results, points


def generate(q1, q2, answer, model_google, options):
    sentences = []
    for i in options:
        sentences.append(q1 + answer[i] + q2)
    sentences = Word2Vec.cleanText(sentences)
    n_dim = 300
    vectors = [Word2Vec.buildWordVector(model_google, z, n_dim) for z in sentences]
    dataset = []
    for a in vectors:
        sentence = np.zeros((49, 300))
        m = len(a)
        start = int((49 - m) / 2)
        sentence[start:start + m] = a
        dataset.append(np.array(sentence))

    question = []
    for i in options:
        question.append(q1 + q2)
    question = Word2Vec.cleanText(question)
    n_dim = 300
    q = [Word2Vec.buildWordVector(model_google, z, n_dim) for z in question]
    q_set = []
    for a in q:
        sentence = np.zeros((49, 300))
        m = len(a)
        start = int((49 - m) / 2)
        sentence[start:start + m] = a
        q_set.append(np.array(sentence))

    option = []
    for i in options:
        option.append(answer[i])
    option = Word2Vec.cleanText(option)
    n_dim = 300
    a = [Word2Vec.buildWordVector(model_google, z, n_dim) for z in option]
    a_set = []
    for a in a:
        sentence = np.zeros((4, 300))
        m = len(a)
        if not m == 0:
            start = int((4 - m) / 2)
            sentence[start:start + m] = a
        a_set.append(np.array(sentence))
    return dataset, q_set, a_set


if __name__ == "__main__":
    connection = pymysql.connect(user='root', password='root', database='GRE')
    cursor = connection.cursor()
    print "Loading models......"
    model_google = gensim.models.KeyedVectors.load_word2vec_format('../GoogleModel/GoogleNews-vectors-negative300.bin', binary=True)
    print "Loading Google Model Finished."
    my_network = model2.Attensive_Reader(name="TC")
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "../test4/model.ckpt")
        print "Loading LSTM Model and opening Tensorflow Finished."
        count = 0
        number = 1001
        for No in range(number):
            print "========== No: " + str(No + 1) + " =========="
            commit = "select * from GREQ1 where No=%d" % (No + 1)
            cursor.execute(commit)
            question = cursor.fetchall()[0]
            question1 = question[1]
            question2 = question[2]
            print "Question: " + question1 + "_____" + question2
            commit = "select * from GREA1 where No=%d" % (No + 1)
            cursor.execute(commit)
            option = cursor.fetchall()[0]
            answer = {}
            answer["A"] = option[1]
            answer["B"] = option[2]
            answer["C"] = option[3]
            answer["D"] = option[4]
            answer["E"] = option[5]
            right_answer = option[6]
            options = ["A", "B", "C", "D", "E"]
            print "Options: "
            for i in options:
                print i + ". " + answer[i]
            start = time.clock()
            data, q, a = generate(question1, question2, answer, model_google, options)
            print "Analysis......"
            results, points = predict(my_network, data, q, a, sess)
            distance = []
            if number == 1:
                draw(points, options)
            for i in range(5):
                distance.append(points[i][0][1] - points[i][0][0])
            maximum = max(distance)
            for i in range(5):
                distance[i] /= maximum
                if distance[i] == 1:
                    elapsed = (time.clock() - start)
                    print "Answer: " + options[i] + " Time used: " + str(elapsed) + "s"
                    if options[i] == right_answer:
                        count += 1
            print distance
            print "Right answer: " + right_answer
        print count
        print "Accuracy: " + str(count/float(number))
        connection.close()