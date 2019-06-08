# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, GRU, SimpleRNN, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
import gensim
from gensim.models import KeyedVectors
from sklearn.model_selection import KFold          
from sklearn.metrics import hamming_loss
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from keras.models import Sequential

def extract_label(Y):
  Y1 = []
  Y2 = []
  Y3 = []

  for i in range(len(Y)):
    for j in range(len(Y[i])):
      if j == 0:
        Y1.append(Y[i][j])
      elif j == 1:
        Y2.append(Y[i][j])
      elif j == 2:
        Y3.append(Y[i][j])

  Y1 = np.array(Y1).reshape([-1,1])
  Y2 = np.array(Y2).reshape([-1,1])
  Y3 = np.array(Y3).reshape([-1,1])

  return Y1, Y2, Y3

def combine_label(y1, y2, y3):
  y = np.zeros((len(y1),3))
  for i in range(len(y)):
    for j in range(len(y[i])):
      if j == 0:
        y[i][j] = y1[i]
      elif j == 1:
        y[i][j] = y2[i]
      elif j == 2:
        y[i][j] = y3[i]
        
  return y

def load_pretrained_word2vec(vocab_size, tokenizer):
  model = KeyedVectors.load_word2vec_format('hadis_wc_model300.bin',  binary=True)
#   model = gensim.models.word2vec.Word2Vec.load('gdrive/My Drive/Dataset/Pre-Trained Embedding/id.bin')
  embedding_matrix = np.zeros((vocab_size + 1, 300))
  for word, i in tokenizer.word_index.items():
    try:
      embedding_matrix[i] = model[word]
    except KeyError:
      pass
    
  return embedding_matrix

def rnn_kfold(X,Y):
  kfold = KFold(n_splits=5, shuffle=False)

  accuracy = []

  for train, test in kfold.split(X, Y):
    model1 = simpleRNN()
    model2 = simpleRNN()
    model3 = simpleRNN()
    
    Y1_train,Y2_train,Y3_train = extract_label(Y[train])
    Y1_test,Y2_test,Y3_test = extract_label(Y[test])

    max_words = 4900
    max_len = 175
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X)
    sequences = tok.texts_to_sequences(X[train])
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len, truncating='post')

    model1.fit(sequences_matrix,Y1_train,batch_size=32,epochs=10,
            validation_split=0.0)

    model2.fit(sequences_matrix,Y2_train,batch_size=32,epochs=10,
              validation_split=0.0)

    model3.fit(sequences_matrix,Y3_train,batch_size=32,epochs=10,
              validation_split=0.0)

    test_sequences = tok.texts_to_sequences(X[test])
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

    Y_real = combine_label(Y1_test, Y2_test, Y3_test)

    y1_predict = model1.predict_classes(test_sequences_matrix)
    y2_predict = model2.predict_classes(test_sequences_matrix)
    y3_predict = model3.predict_classes(test_sequences_matrix)

    Y_pred = combine_label(y1_predict, y2_predict, y3_predict)

    accuracy.append(hamming_loss(Y_real, Y_pred))

  return accuracy

def gru_kfold(X,Y):
  kfold = KFold(n_splits=5, shuffle=False)

  accuracy = []

  for train, test in kfold.split(X, Y):
    model1 = gruRNN()
    model2 = gruRNN()
    model3 = gruRNN()
    
    Y1_train,Y2_train,Y3_train = extract_label(Y[train])
    Y1_test,Y2_test,Y3_test = extract_label(Y[test])

    max_words = 4900
    max_len = 175
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X)
    sequences = tok.texts_to_sequences(X[train])
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len, truncating='post')

    model1.fit(sequences_matrix,Y1_train,batch_size=32,epochs=10,
            validation_split=0.0)

    model2.fit(sequences_matrix,Y2_train,batch_size=32,epochs=10,
              validation_split=0.0)

    model3.fit(sequences_matrix,Y3_train,batch_size=32,epochs=10,
              validation_split=0.0)

    test_sequences = tok.texts_to_sequences(X[test])
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

    Y_real = combine_label(Y1_test, Y2_test, Y3_test)

    y1_predict = model1.predict_classes(test_sequences_matrix)
    y2_predict = model2.predict_classes(test_sequences_matrix)
    y3_predict = model3.predict_classes(test_sequences_matrix)

    Y_pred = combine_label(y1_predict, y2_predict, y3_predict)

    accuracy.append(hamming_loss(Y_real, Y_pred))

  return accuracy

def simpleRNN():
    model = Sequential()
    model.add(Embedding(max_words+1, output_dim=300, input_length=max_len, weights=[embedding_matrix]))
    model.add(SpatialDropout1D(0.4))
    model.add(SimpleRNN(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def gruRNN():
    model = Sequential()
    model.add(Embedding(max_words+1, output_dim=300, input_length=max_len, weights=[embedding_matrix]))
    model.add(SpatialDropout1D(0.4))
    model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

df = pd.read_excel('Multilabel Clean.xlsx')

X = df.Data.values
Y1 = df.Anjuran
Y2 = df.Larangan
Y3 = df.Informasi
le = LabelEncoder()

# Y1 = le.fit_transform(Y1)
# Y1 = Y1.reshape(-1,1)
Y1 = Y1.values.reshape([-1,1])

# Y2 = le.fit_transform(Y2)
# Y2 = Y2.reshape(-1,1)
Y2 = Y2.values.reshape([-1,1])

# Y3 = le.fit_transform(Y3)
# Y3 = Y3.reshape(-1,1)
Y3 = Y3.values.reshape([-1,1])

Y = np.zeros((len(Y1),3))
for i in range(len(Y)):
  for j in range(len(Y[i])):
    if j == 0:
      Y[i][j] = Y1[i]
    elif j == 1:
      Y[i][j] = Y2[i]
    elif j == 2:
      Y[i][j] = Y3[i]


max_words = 4900
max_len = 175
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X)
embedding_matrix = load_pretrained_word2vec(max_words, tok)

accuracy = gru_kfold(X,Y)

# rnn 5 fold
# 0.10857253569405945

# gru 5 fold
# 0.10511043548242868