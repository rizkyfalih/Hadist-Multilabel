# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:53:37 2019

@author: rizkyfalih
"""

# Import the libraries
from keras.layers import LSTM, Dense, Embedding, GRU, SimpleRNN, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence      
from sklearn.metrics import hamming_loss
from keras.models import load_model
from keras.models import Sequential
import pandas as pd 
import numpy as np
from preprocess import extract_label, combine_label

def simpleRNN(max_words, max_len, embedding_matrix):
    model = Sequential()
    model.add(Embedding(max_words+1, output_dim=300, input_length=max_len, weights=[embedding_matrix]))
    model.add(SpatialDropout1D(0.4))
    model.add(SimpleRNN(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def lstmRNN(max_words, max_len, embedding_matrix):
    model = Sequential()
    model.add(Embedding(max_words + 1, output_dim=300, input_length=max_len, weights=[embedding_matrix]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def gruRNN(max_words, max_len, embedding_matrix):
    model = Sequential()
    model.add(Embedding(max_words+1, output_dim=300, input_length=max_len, weights=[embedding_matrix]))
    model.add(SpatialDropout1D(0.4))
    model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def rnn_kfold(X, Y, max_words, max_len, tok, embedding_matrix, fold):
  accuracy = []
  listPred = []
  iteration = 0
  for i in range(1,fold+1):
    iteration += 1
    model1 = simpleRNN(max_words, max_len, embedding_matrix)
    model2 = simpleRNN(max_words, max_len, embedding_matrix)
    model3 = simpleRNN(max_words, max_len, embedding_matrix)
    
    # load X_train
    X_train = pd.read_csv("kfold/X_train/fold-"+ str(i) + ".csv")
    X_train = X_train.Data.values
    
    # load X_test
    X_test = pd.read_csv("kfold/X_test/fold-"+ str(i) + ".csv")
    X_test = X_test.Data.values
    
    # load Y_train
    Y_train = np.genfromtxt("kfold/Y_train/fold-"+ str(i) + ".csv", delimiter=',')
    
    # load Y_test
    Y_test = np.genfromtxt("kfold/Y_test/fold-"+ str(i) + ".csv", delimiter=',')
    
    # Extract Y_train and Y_test value
    Y1_train,Y2_train,Y3_train = extract_label(Y_train)
    Y1_test,Y2_test,Y3_test = extract_label(Y_test)

    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len, truncating='post')
    
    model1.fit(sequences_matrix,Y1_train,batch_size=32,epochs=10,
            validation_split=0.0)

    model2.fit(sequences_matrix,Y2_train,batch_size=32,epochs=10,
              validation_split=0.0)

    model3.fit(sequences_matrix,Y3_train,batch_size=32,epochs=10,
              validation_split=0.0)
    
    # Save the models
    model1.save("rnn_kfold/fold_"+ str(iteration) + "-modelRNN1.h5")
    model2.save("rnn_kfold/fold_"+ str(iteration) + "-modelRNN2.h5")
    model3.save("rnn_kfold/fold_"+ str(iteration) + "-modelRNN3.h5")

    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

    Y_real = combine_label(Y1_test, Y2_test, Y3_test)

    y1_predict = model1.predict_classes(test_sequences_matrix)
    y2_predict = model2.predict_classes(test_sequences_matrix)
    y3_predict = model3.predict_classes(test_sequences_matrix)

    Y_pred = combine_label(y1_predict, y2_predict, y3_predict)
    listPred.append(Y_pred)

    accuracy.append(hamming_loss(Y_real, Y_pred))

  return accuracy, listPred