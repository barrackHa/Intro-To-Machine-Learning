#!/usr/bin/env python
# coding: utf-8


import sys
sys.path.append("../../")
from utils import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class Perceptron():
    def __init__(self):
        self.model = None
    
    def fit(self,X,y):
        w = np.zeros(1+X.shape[1])
        w[0] = 1
        X = np.column_stack((np.ones(X.shape[0]),X))
        exists_y_i = True
        tmp = X.dot(w.T)
        while exists_y_i:
            tmp = y*np.inner(X,w)
            for i, val in enumerate(tmp):
                if val <= 0:
                    # update w
                    w = w + y[i]*X[i]
                    tmp = X.dot(w.T)
                    continue
                # else all i's hold y[i]*np.inner(X[i],w) > 0
                exists_y_i = False
        self.model = w
        print(self.model)
        return 
    
    def predict(self,X):
        pass
    
    def score(self,X,y):
        pass
    
class SVM():
    def __init__(self):
        self._svm = SVC(C=1e10, kernel="linear")
        
    def fit(self,X,y):
        self._svm = self._svm.fit(X,y)
    
    def predict(self,X):
        return self._svm.predict(X)
    
    def score(self,X,y):
        pass

class Logistic():
    def __init__(self):
        self._logistic = LogisticRegression(solver="liblinear")
     
    def fit(self,X,y):
        self._logistic = self._logistic.fit(X,y)
         
    def predict(self,X):
        return self._logistic.predict(X)
    
    def score(self,X,y):
        pass



