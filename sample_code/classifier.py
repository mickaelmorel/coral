from sklearn.base import BaseEstimator
from sklearn import tree
from data_manager import DataManager # Classe du binôme 1
from preprocessor import Preprocessor # Classe du binöme 2

import pickle


class Classifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.clf =  DecisionTreeClassifier()
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes
        
    def get_classes(self):
        return self.clf.classes_
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
