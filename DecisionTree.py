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


# Convert the options from binary format to an integer
def getOptionValue(options):   
    optionlist = []
    for i in range(len(options)):
        optionlist.append("0")
    for i, option in enumerate(options):
        if options[option] == True:
            optionlist[i] = "1"                      
    return int(''.join(optionlist), 2)  
   
def parseJson(json_file):
    businesses = []    
    with open(json_file, 'r') as f:
        for line in f:
            businesses.append(loads(line))
         
    data = []
    success_count = 0
    i = 1
    for business in businesses:       
        if ('Restaurants' in business['categories']) and (business['review_count'] > 35):# and (business['city'] == 'Las Vegas'): #'Phoenix'
            stars = int(business['stars'])
            success = 1 if business['stars'] > 3.5 else 0
            if success == 1:
                success_count += 1
            attributes = business['attributes']
            keys =["Price Range", "Attire", "Alcohol", "Noise Level", "Smoking", "Wi-Fi", "Has TV",\
                   "Accepts Credit Cards", "Take-out", "Happy Hour", "Outdoor Seating", "Takes Reservations", \
                   "Waiter Service", "Caters", "Good For Kids", "Good For Groups", "Wheelchair Accessible", "Delivery",\
                   "Ages Allowed", "BYOB/Corkage", "Parking", "Good For", "Ambience"]
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
                    if attributes[key] == "beer_and_wine":
                        features[key] = 1
                    elif attributes[key] == "full_bar":
                        features[key] = 2
                    else:
                        features[key] = 0
                elif key == "Noise Level":
                    if (attributes[key] == "none"):
                        features[key] = 0
                    elif (attributes[key] == "average"):                   
                        features[key] = 1
                    else:
                        features[key] = 2              
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
                    or key == "Good For Groups" or key == "Wheelchair Accessible" or key == "Delivery"):
                
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
                elif key == "Parking" or key == "Good For" or key == "Ambience":                 
                    features[key] = getOptionValue(attributes[key])                             
            data.append((features, success))#stars))
        
    print("Successful Restaurants: " + str(success_count))
    print("Total Restaurants: " + str(len(data)))
    data_file = open('vegas_restaurant_data.json','w')    
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
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)   
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
                #weights[k] -=  eta * gradient_loss
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
    plt.bar(Index, values)
    plt.xticks(Index, names, rotation=degree)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()


data = parseJson("yelp_academic_dataset_business.json")


random.shuffle(data)
half = len(data)/2
trainExamples = data[:half]
testExamples = data[half:]
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


X_input = SelectKBest(chi2, k=23).fit_transform(X_input, Y_input)
X_train, X_test, Y_train, Y_test = train_test_split(X_input, Y_input, test_size=0.4, random_state=42)


Classifiers = [
    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),
    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200),
    AdaBoostClassifier(),
    GaussianNB()]


Accuracy=[]
Model=[]
for classifier in Classifiers:    
    fit = classifier.fit(X_train, Y_train)
    pred = fit.predict(X_test)   
    accuracy = accuracy_score(pred, Y_test)
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))


compBarChart('Model', 'Accuracy', 'Accuracies of Models', Model, Accuracy, 45)


rfecv = RFECV(estimator=Classifiers[5], step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X_train, Y_train)


print("Optimal number of features : %d" % rfecv.n_features_)


# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()




