import tensorflow as tf
import model1
import gensim
import Word2Vec
import numpy as np
import pymysql


def predict(model, data, sess):
    results = []
    points = []
    for batch_xs in data:
        batch_xs = batch_xs.reshape((model.batch_size, model.steps, model.inputs))
        results.append(sess.run(tf.argmax(model.output, 1), feed_dict={model.x: batch_xs, model.keep_prob: 1.0}))
        points.append((sess.run(model.output, feed_dict={model.x: batch_xs, model.keep_prob: 1.0})))
    return results, points


def storeVecs(input, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(input, fw)
    fw.close()


def grabVecs(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


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
    return dataset


if __name__ == "__main__":
    connection = pymysql.connect(user='root', password='root', database='GRE')
    cursor = connection.cursor()
    print "Loading models......"
    model_google = gensim.models.KeyedVectors.load_word2vec_format('./GoogleModel/GoogleNews-vectors-negative300.bin', binary=True)
    print "Loading Google Model Finished."
    my_network = model1.Bd_LSTM_layer(name="TC")
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./model/model.ckpt")
        print "Loading LSTM Model and opening Tensorflow Finished."
        count = 0
        for No in range(100):
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

            data = generate(question1, question2, answer, model_google, options)
            # data =grabVecs("predict.pkl")
            print "Analysis......"
            results, points = predict(my_network, data, sess)
            distance = []
            for i in range(5):
                distance.append(points[i][0][1] - points[i][0][0])
            maximum = max(distance)
            for i in range(5):
                distance[i] /= maximum
                if distance[i] == 1:
                    print "Answer: " + options[i]
                    if options[i] == right_answer:
                        count += 1
            print distance
            print "Right answer: " + right_answer
        print "Accuracy: " + str(count/100.0)
