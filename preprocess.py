# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:59:44 2019

@author: rizkyfalih
"""

import gensim
from gensim.models import KeyedVectors
import numpy as np

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

def initializeLabel(y):
  init = []
  for i in range(len(y)):
    if(y[i][0] == 0 and y[i][1] == 0 and y[i][2] == 0):
      init.append('1')
    elif (y[i][0] == 0 and y[i][1] == 0 and y[i][2] == 1):
      init.append('2')
    elif (y[i][0] == 0 and y[i][1] == 1 and y[i][2] == 0):
      init.append('3')
    elif (y[i][0] == 1 and y[i][1] == 0 and y[i][2] == 0):
      init.append('4')
    elif (y[i][0] == 0 and y[i][1] == 1 and y[i][2] == 1):
      init.append('5')
    elif (y[i][0] == 1 and y[i][1] == 0 and y[i][2] == 1):
      init.append('6')
    elif (y[i][0] == 1 and y[i][1] == 1 and y[i][2] == 0):
      init.append('7')
    elif (y[i][0] == 1 and y[i][1] == 1 and y[i][2] == 1):
      init.append('8')
  return init

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