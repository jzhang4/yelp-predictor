import random
import collections
import math
import sys
from collections import Counter
from util import *
from json import loads
from json import dumps
from re import sub
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model

"""
Returns true if a file ends in .json
"""
def isJson(f):
    return len(f) > 5 and f[-5:] == '.json'

def getAccuracy(y, predict):
    return sum([1 for i in zip(y, predict) if i[0]!=i[1]]) * 1.0 / len(y)

def neuralPredict(json_file):
    # getting features into a map
    trainData, testData = extractJSONDict(json_file)
    trainx, trainy = zip(*trainData)
    testx, testy = zip(*testData)
    trainx, trainy, testx, testy = list(trainx), list(trainy), list(testx), list(testy)
    print 'Training on', len(trainx), 'samples'

    clf = MLPClassifier()
    clf.fit(trainx, trainy)
    predictions = clf.predict(testx)
    print 'Accuracy:', getAccuracy(testy, predictions)

    l = linear_model.LogisticRegression()
    l.fit(trainx, trainy)
    predictions = l.predict(testx)
    print 'Accuracy:', getAccuracy(testy, predictions)

def main(argv):
    f = 'yelp-data/restaurant_data_neural.json'
    if isJson(f):
        neuralPredict(f)

if __name__ == '__main__':
    main(sys.argv)
