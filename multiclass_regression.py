import random
import collections
import math
import sys
from collections import Counter
from util import *
from json import loads
from json import dumps
from re import sub
from sklearn import linear_model
from sklearn import metrics
import numpy as np


"""
Returns true if a file ends in .json
"""
def isJson(f):
    return len(f) > 5 and f[-5:] == '.json'

def learnPredictor(json_file):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    # getting features into a map
    trainExamples, testExamples = extractJSONArray(json_file)
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    featuresList, valuesList = zip(*trainExamples)
    nonContinuousValues = [value*2 for value in valuesList]
    #valuesList = map(long, valuesList)
    logreg = linear_model.LogisticRegression(C=.1, multi_class='multinomial', solver='lbfgs')
    logreg = logreg.fit(featuresList, nonContinuousValues)
    expected = nonContinuousValues
    predicted = logreg.predict(featuresList)
    print(metrics.classification_report(expected, predicted))

    testFeatures, testValues = zip(*testExamples)
    nonContinuousTestValues = [value*2 for value in testValues]
    print(metrics.classification_report(nonContinuousTestValues, logreg.predict(testFeatures)))

    # END_YOUR_CODE


def main(argv):
    f = "restaurant_data_multiclass.json"
    if isJson(f):
        learnPredictor(f)

if __name__ == '__main__':
    main(sys.argv)
