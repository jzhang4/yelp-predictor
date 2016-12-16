import sys
import random
from json import loads
from json import dumps
from re import sub
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.neural_network import MLPClassifier


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from util import *
import datetime
import argparse

# Convert the options from binary format to an integer
def getOptionValue(options):  
    if options == {}:
        return 0

    optionlist = []
    for i in range(len(options)):
        optionlist.append("0")
    for i, option in enumerate(options):
        if options[option] == True:
            optionlist[i] = "1"  
    return int(''.join(optionlist), 2)  

def parseJson(json_file, review_count=35, binary_classify=True, city = ""):   
    businesses = [] 
    with open(json_file, 'r') as f:
        for line in f:
            businesses.append(loads(line))
    data = []
    success_count = 0
    i = 1
    for business in businesses: 
        if ('Restaurants' in business['categories']) and (business['review_count'] > review_count)\
                and (city == "" or business['city'] == city):
            if binary_classify:
                y = 1 if business['stars'] > 3.5 else 0
                if y == 1:
                    success_count += 1
            else:
                y = int(business['stars'])

            attributes = business['attributes']
            keys =["Alcohol", "Noise Level", "Has TV", "Attire", "Ambience", "Good For Kids", "Price Range",
                   "Good For Dancing", "Delivery", "Dogs Allowed", "Coat Check", "Smoking", "Accepts Credit Cards",
                   "Take-out", "Happy Hour", "Wheelchair Accessible", "Outdoor Seating", "Takes Reservations",
                   "Waiter Service", "Wi-Fi", "Drive-Thru", "Caters", "Good For", "Parking", "Music", "Good For Groups",
                   "Ages Allowed", "BYOB", "BYOB/Corkage", "Corkage", "Order at Counter", "Open 24 Hours", "Dietary Restrictions"]
            features = {}
            for key in keys:
               features[key] = 0
            for key in attributes:
                if key == "Price Range":
                    features[key] = attributes[key]
                elif key == "Attire":
                    if attributes[key] == "casual": 
                        features[key] = 1
                    elif attributes[key] == "dressy":
                        features[key] = 2
                    else:
                        features[key] = 0
                elif key == "Alcohol":
                    if (attributes[key] == "none"):
                        features[key] = 0
                    elif attributes[key] == "beer_and_wine":
                        features[key] = 1
                    elif attributes[key] == "full_bar":
                        features[key] = 2
                elif key == "Noise Level":
                    if (attributes[key] == "none"):
                        features[key] = 0
                    elif attributes[key] == "average":
                        features[key] = 1
                    elif attributes[key] == "loud":
                        features[key] = 2
                    else:
                        features[key] = 3
                elif key == "Smoking":
                    if attributes[key] == "no":
                        features[key] = 0
                    else:
                        features[key] = 1
                elif key == "Wi-Fi":
                    if attributes[key] == "free":
                        features[key] = 1
                    elif attributes[key] == "paid":
                        features[key] = 2
                    else:
                        features[key] = 0
                elif (key == "Has TV" or key =="Accepts Credit Cards" or key == "Take-out" \
                    or key == "Happy Hour" or key == "Outdoor Seating" or key == "Takes Reservations" \
                    or key == "Waiter Service" or key == "Caters" or key == "Good For Kids" \
                    or key == "Good For Groups" or key == "Wheelchair Accessible" or key == "Delivery"\
                    or key == "Good For Dancing" or key =="Dogs Allowed" or key == "Coat Check"
                    or key == "Drive-Thru" or key == "Open 24 Hours" or key == "Order at Counter"
                    or key == "BYOB" or key == "Corkage"):

                    if attributes[key] == True:
                        features[key] = 1
                    else:
                        features[key] = 0

                elif key == "Ages Allowed":
                    if attributes[key] == "18plus":
                        features[key] = 1
                    elif attributes[key] == "21plus":
                        features[key] = 2
                    else:
                        features[key] = 0
                elif key == "BYOB/Corkage":
                    if attributes[key] == "yes_free":
                        features[key] = 1
                    elif attributes[key] == "yes_corkage":
                        features[key] = 2
                    else:
                        features[key] = 0
                elif key == "Parking" or key == "Good For" or key == "Ambience" or key == "Music" or key == "Dietary Restrictions": 

                    features[key] = getOptionValue(attributes[key])
            data.append((features, y))
    if binary_classify == True:
        print("Successful Restaurants: " + str(success_count))
    print("Total Restaurants: " + str(len(data)))
    data_file = open('yelp-data/vegas_restaurant_data.json','w')
    for business in data:
        data_file.write(dumps(business, data_file))
        data_file.write('\n')
    return data

def learnPredictor(trainExamples, testExamples, numIters, eta):
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
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution iplt.plots 12 lines of code, but don't worry if you deviate from this)   
    for i in range(0, numIters):
        for trainExample in trainExamples:
            x = trainExample[0]
            y = trainExample[1]
            features = x
            for k in features:
                if k not in weights:
                    weights[k] = 0
            d = 1 - sum(weights[k] * features[k] for k in features) * y
            for k in features:
                if d > 0:
                    gradient_loss = - features[k] * y
                else:
                    gradient_loss = 0
                weights[k] = weights[k] - eta * gradient_loss
        predictor = lambda(x) : (1 if dotProduct(x, weights) >= 0 else -1)
        trainLoss = evaluatePredictor(trainExamples, predictor)
        testLoss = evaluatePredictor(testExamples, predictor)
        print "Iteration %d: training error = %f, test error = %f " % (i, trainLoss, testLoss)
    # END_YOUR_CODE
    return weights

def compBarChart(xlabel, ylabel, title, names, values, degree):
    plt.figure()
    Index = []
    for i in range(len(values)):
        Index.append(i+1)
    plt.bar(Index, values, color = 'coral')
    plt.xticks(Index, names, rotation=degree)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

# Plot number of features VS. cross-validation scores
def plotPerfScore(rfecv):
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='coral')
    plt.title('Cross-Validation Scores of Best Chi-Squared Features')
    plt.show()

def classification(json_file, review_count=35, binary_classify = True, city = "", plot = False):
    data = parseJson(json_file, review_count, binary_classify, city)
    if binary_classify == True: #baseline model using SGD
        train_size = (int)(0.6*len(data))
        trainExamples = data[:train_size]
        testExamples = data[train_size:]
        weights = learnPredictor(trainExamples, testExamples, numIters=20, eta=0.01)
        print "Weights = ", weights

        example = data[0]
        features = example[0]
        names = []
        for feature in features:
            names.append(feature)
        values = []
        for key in weights:
            values.append(weights[key])
        if plot:
            compBarChart('Feature', 'Weight', 'Weights of Features', names, values, 90)

    X_input = []
    Y_input = []
    for example in data:
        features = example[0]
        list =[]
        for key in features:
            list.append(features[key])
        X_input.append(list)
        Y_input.append(example[1])
    X_input = SelectKBest(chi2, k=len(features)).fit_transform(X_input, Y_input)
    X_train, X_test, Y_train, Y_test = train_test_split(X_input, Y_input, test_size=0.4, random_state=42)

    if binary_classify == True:
        logistic_regression = LogisticRegression(C=0.1,solver='lbfgs')
        kmean_classifier = KNeighborsClassifier(2)
    else:
        logistic_regression = LogisticRegression(C=0.1, multi_class='multinomial', solver='lbfgs')
        kmean_classifier = KNeighborsClassifier(9)
    Classifiers = [
        logistic_regression,
        MLPClassifier(solver='lbfgs'),
        kmean_classifier,
        SVC(kernel="rbf", C=0.025, probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GaussianNB()]

    Accuracy=[]
    Model=[]
    for classifier in Classifiers:
        a = datetime.datetime.now()
        fit = classifier.fit(X_train, Y_train)
        testpred = fit.predict(X_test)
        b = datetime.datetime.now()
        trainpred = fit.predict(X_train)
        trainaccuracy = accuracy_score(trainpred, Y_train)
        testaccuracy = accuracy_score(testpred, Y_test)
        Accuracy.append(testaccuracy)
        Model.append(classifier.__class__.__name__)
        print('Train accuracy of '+classifier.__class__.__name__+' is '+str(trainaccuracy))
        print('Test accuracy of '+classifier.__class__.__name__+' is '+str(testaccuracy))
        print('Elapsed time = '+ str(b-a))
    if binary_classify == True:
        model = "Binary Model"
    else:
        model = "Multiclass Model"
    if plot:
        compBarChart(model, 'Accuracy', 'Accuracies of Models', Model, Accuracy, 45)

    print("Starting to calculate the cross-validation scores for each model...")

    for classifier in Classifiers:
        try:
            rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(2), scoring='accuracy')
            rfecv.fit(X_train, Y_train)
            print("Optimal number of features for " +classifier.__class__.__name__+ ": %d" % rfecv.n_features_)
            if plot:
                plotPerfScore(rfecv)
        except:
            pass

def main(argv):
    parser = argparse.ArgumentParser(description='Classification for yelp restaurants.')
    classification("yelp-data/yelp_academic_dataset_business.json", 35, binary_classify = False, plot = False)

if __name__ == '__main__':
    main(sys.argv)
