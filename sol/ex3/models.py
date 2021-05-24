#!/usr/bin/env python

import sys
sys.path.append("../../")
from utils import *

from sklearn.linear_model import Perceptron as SkPerceptron
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class Perceptron():
    def __init__(self):
        self.model = None
        self.__sk_model = None
    
    @property
    def weights(self):
        return self.model
    
    @property
    def sk_weights(self):
        return np.append(self.__sk_model.intercept_, self.__sk_model.coef_[0])
    
    def fit(self,X,y):
        x = np.insert(X,0,np.ones(X.shape[0]).T,1)
    
        model = np.zeros(x.shape[1])
        w = np.zeros_like(y)
        go = (y * w) <= 0

        while go.any():
            corr_idx = np.where(go)[0][0]
            model += (y[corr_idx] * x[corr_idx]).reshape(model.shape)
            w = np.dot(x, model)
            go = np.squeeze((y * w) <= 0)
        self.model = model
        
        
    def sk_fit(self,X,y):
        p = SkPerceptron(fit_intercept=True)
        p.fit(X,y)
        self.__sk_model = p
    
    def __predict_vector__(self,v):
        return np.sign( np.dot(v, self.model) )
    
    def predict(self,X, use_sk=False):
        if use_sk:
            return self.__sk_model.predict(X)
        
        y =[]
        for v in X:
            y.append(self.__predict_vector__(v))
        return y
        
    
    def score(self,X,y):
        y_test = self.predict(X,y)
        num_samples = X.shape[0]
        num_of_errors =  sum(np.not_equal(self.predict(X,y),y))
        error = num_of_errors / num_samples
        accuracy = (num_samples - num_of_errors) / num_samples
        true_pos, false_pos  = 0, 0
        for i in len(y):
            if y_test[i] == 1:
                true_pos += int(y[i] == y_test[i])
                false_pos += int(y[i] != y_test[i])
        return {
            'num_samples':  num_samples,
            'error':  error,
            'accuracy':  accuracy,
            'FPR': false_pos,
            'TPR': true_pos,
            'precision':  'precision',
            'specificty':  'specificty'
        }
class SVM():
    def __init__(self):
        self._svm = SVC(C=1e10, kernel="linear")
        
    def fit(self,X,y):
        self._svm = self._svm.fit(X,y)
        print(self._svm.get_params())
    
    def predict(self,X):
        return self._svm.predict(X)
    
    def score(self,X,y):
        pass
    
    
class Logistic():
    def __init__(self):
        self._logistic = LogisticRegression(solver="liblinear")
     
    def fit(self,X,y):
        self._logistic = self._logistic.fit(X,y)
        print(self._logistic.get_params())
         
    def predict(self,X):
        return self._logistic.predict(X)
    
    def score(self,X,y):
        pass



