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

def neuralParseJson(json_file):
    data = []
    with open(json_file, 'r') as f:
        for line in f:
            data.append(loads(line))
    indexes = [{},0]
    for d in data:
        for f in d[0]:
            if f not in indexes[0]:
                indexes[0][f] = indexes[1]
                indexes[1] += 1

    results = []
    for d in data:
        features = [0]*indexes[1]
        for f in d[0]:
            features[indexes[0][f]] = d[0][f]
        results.append((features, d[1]))

    with open('yelp-data/restaurant_data_neural.json','w') as data_file:
        for r in results:
            data_file.write(dumps(r, data_file))
            data_file.write('\n')

    with open('yelp-data/features_indices.txt','w') as feature_index_file:
        feature_index_file.write(dumps(indexes[0], feature_index_file))

def main(argv):
    f = "yelp-data/restaurant_data.json"
    if isJson(f):
        neuralParseJson(f)

if __name__ == '__main__':
    main(sys.argv)
