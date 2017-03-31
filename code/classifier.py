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
    def __init__(self):
        '''This method initializes the parameters. This is where you could replace
        RandomForestClassifier by something else or provide arguments, e.g.
        RandomForestClassifier(n_estimators=100, max_depth=2)'''
        self.clf = RandomForestClassifier()

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
        
    def get_classes(self):
        return self.clf.classes_
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
return self

class Classifier(BaseEstimator):
    def __init__(self):
        fancy_classifier = Pipeline([
                ('preprocessing', Preprocessor()),
                ('classification', BaggingClassifier(base_estimator=RandomForestClassifier()))
                ])
        self.clf = VotingClassifier(estimators=[
                ('basic', BasicClassifier()), 
                ('fancy', fancy_classifier)], 
                voting='soft')   
        
    def fit(self, X, y):
        return self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
