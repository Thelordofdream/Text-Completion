import numpy as np
import pymysql.cursors


def grabVecs(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


def storeVecs(input, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(input, fw)
    fw.close()

goole_vecs = grabVecs('../data for input1/q_vecs.pkl')
dataset = []
for a in goole_vecs:
    sentence = np.zeros((49, 300))
    m = len(a)
    start = int((49 - m) / 2)
    sentence[start:start + m] = a
    dataset.append(sentence)
storeVecs(dataset, '../data for input0/q.pkl')

goole_vecs = grabVecs('../data for input1/a_vecs.pkl')
dataset = []
for a in goole_vecs:
    sentence = np.zeros((4, 300))
    m = len(a)
    if not m == 0:
        start = int((4 - m) / 2)
        sentence[start:start + m] = a
    dataset.append(sentence)
storeVecs(dataset, '../data for input0/a.pkl')