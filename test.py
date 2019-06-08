# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:05:52 2019

@author: rizkyfalih
"""

import pandas as pd
from preprocess import combine_label
from keras.preprocessing.text import Tokenizer
from preprocess import load_pretrained_word2vec
from classifier import rnn_kfold


df = pd.read_excel('Multilabel Clean.xlsx')

X = df.Data.values
Y1 = df.Anjuran
Y2 = df.Larangan
Y3 = df.Informasi

Y1 = Y1.values.reshape([-1,1])
Y2 = Y2.values.reshape([-1,1])
Y3 = Y3.values.reshape([-1,1])
      
Y = combine_label(Y1, Y2, Y3)

max_words = 8000
max_len = 175
tok = Tokenizer(num_words=max_words, lower=False)    
tok.fit_on_texts(X)
embedding_matrix = load_pretrained_word2vec(max_words, tok)

accuracy, pred = rnn_kfold(X, Y, max_words, max_len, tok, embedding_matrix, 5) 