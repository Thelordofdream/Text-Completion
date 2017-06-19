import gensim
import pymysql.cursors

import Word2Vec

# ===========================================
# load data
connection = pymysql.connect(user='root', password='root', database='GRE')
cursor = connection.cursor()
commit = "select * from GRES"
cursor.execute(commit)
Sentences = [each[1] for each in cursor.fetchall()]
Sentences = Word2Vec.cleanText(Sentences)

# ===========================================
# Load model
model_google = gensim.models.KeyedVectors.load_word2vec_format('../GoogleModel/GoogleNews-vectors-negative300.bin', binary=True)
# Word2Vec.Train_Wrod2VEc(Sentences, model_google)

# ===========================================
# Generalize words
n_dim = 300
train_vectors = [Word2Vec.buildWordVector(model_google, z, n_dim) for z in Sentences]
Word2Vec.storeVecs(train_vectors, '../data for input/word_vecs.pkl')
