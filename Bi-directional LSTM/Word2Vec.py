import gensim
import numpy as np

LabeledSentence = gensim.models.doc2vec.LabeledSentence


# ===========================================
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    # treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, '') for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


def buildWordVector(model_w2v, text, size):
    sen = []
    vec = np.zeros(size).reshape((1, size))
    for word in text:
        try:
            vec = model_w2v[word].reshape((1, size))
            sen.extend(vec)
        except:
            continue
    return sen


def storeVecs(input, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(input, fw)
    fw.close()


def Train_Wrod2VEc(Sentences, model_w2v):
    # ===========================================
    # Train the model over train_reviews (this may take several minutes)
    model_w2v.train(Sentences)

    # ===========================================
    # Store model and result
    model_w2v.save("../model/model_w2v")