# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:37:45 2019

@author: rizkyfalih
"""

from sklearn.model_selection import KFold 
from preprocess import extract_label
import pandas as pd 
import numpy as np

def split_kfold(X, Y, fold):
  kfold = KFold(n_splits=fold, shuffle=False)
  iteration = 0
  for train, test in kfold.split(X, Y):
    iteration += 1
    Y1_train,Y2_train,Y3_train = extract_label(Y[train])
    Y1_test,Y2_test,Y3_test = extract_label(Y[test])
    
    # save X_train
    sampleX_train = pd.DataFrame(X[train], columns=['Data']) 
    sampleX_train.to_csv("kfold/X_train/fold-"+ str(iteration) + ".csv", index = None)

    # save X_test
    sampleX_test = pd.DataFrame(X[test], columns=['Data']) 
    sampleX_test.to_csv("kfold/X_test/fold-"+ str(iteration) + ".csv", index = None)

    # save Y_train
    np.savetxt("kfold/Y_train/fold-"+ str(iteration) + ".csv", Y[train], delimiter=",")

    # save Y_test
    np.savetxt("kfold/Y_test/fold-"+ str(iteration) + ".csv", Y[test], delimiter=",")
  
  print("kfold split finished")