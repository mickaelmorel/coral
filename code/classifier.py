import numpy as np
from sys import argv
from sklearn.base import BaseEstimator
from PreProcessor import Preprocessor

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle

def ebar(score, sample_num):
    '''ebar calculates the error bar for the classification score (accuracy or error rate)
    for sample_num examples'''
return np.sqrt(1.*score*(1-score)/sample_num)

class BasicClassifier(BaseEstimator):
    '''BasicClassifier: modify this class to create a simple classifier of
    your choice. This could be your own algorithm, of one for the scikit-learn
    classfiers, with a given choice of hyper-parameters.'''
    def __init__(self):
        '''This method initializes the parameters. This is where you could replace
        RandomForestClassifier by something else or provide arguments, e.g.
        RandomForestClassifier(n_estimators=100, max_depth=2)'''
        self.clf = RandomForestClassifier()

    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        return self.clf.fit(X, y)

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.clf.predict(X)

    def predict_proba(self, X):
        ''' Similar to predict, but probabilities of belonging to a class are output.'''
        return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes
        
    def get_classes(self):
        return self.clf.classes_
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
return self
