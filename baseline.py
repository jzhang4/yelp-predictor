import random
import collections
import math
import sys
from collections import Counter
from util import *
from json import loads
from json import dumps
from re import sub


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
    examples = []
    with open(json_file, 'r') as f:
        for line in f:
            examples.append(loads(line))
    
    random.shuffle(examples)
    half = len(examples)/2
    trainExamples = examples[:half]
    testExamples = examples[half:]

    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    numIters=20
    eta=0.01
    weights = {}  # feature => weight

    for i in xrange(numIters):
        for trainExample in trainExamples:
            features = trainExample[0]
            dot_product = dotProduct(features, weights)
            margin = dot_product * trainExample[1]

            if margin < 1:
                for key in features:
                    update = eta * features[key] * trainExample[1]
                    if key in weights:
                        weights[key] += update
                    else: 
                        weights[key] = update

        print("Train Error Iter " + str(i) + ": " + str(evaluatePredictor(trainExamples, lambda(x) : (1 if dotProduct(x, weights) >= 0 else -1))))
        print("Test Error  Iter " + str(i) + ": " + str(evaluatePredictor(testExamples, lambda(x) : (1 if dotProduct(x, weights) >= 0 else -1))))
        print("\n")
    # END_YOUR_CODE
    return weights


def main(argv):
    f = "restaurant_data.json"
    if isJson(f):
        learnPredictor(f)

if __name__ == '__main__':
    main(sys.argv)
