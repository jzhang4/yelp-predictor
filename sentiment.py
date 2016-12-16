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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from util import *
import datetime
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from optparse import OptionParser
from collections import defaultdict




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


def parseReviews(business_file, review_file, review_count=35, binary_classify=True, cities = []):
    business_stars = {}   
    with open(business_file, 'r') as f:
        for line in f:
            business = loads(line)
            if ('Restaurants' in business['categories']) and (business['review_count'] > review_count)\
                and (cities == [] or business['city'] in cities):
                id = business['business_id']                               
                if binary_classify == True:
                    y = 1 if business['stars'] > 3.5 else 0
                else:
                    y = int(business['stars'])
                business_stars[id] = y
                
    print len(business_stars)
    
    X = []
    Y = []
    texts = {}
    print "Processing review file..."   
    
    with open(review_file, 'r') as f:
        for line in f:
            review = loads(line)
            id = review['business_id']
            if id in business_stars:
                if id in texts:
                    texts[id] = ' '.join([texts[id], review['text']])
                else:
                    texts[id] = review['text']                                   
    list = []   
    for id in texts:
        list.append(texts[id])
        Y.append(business_stars[id])
    print len(list), len(Y)
    print "TfidfVectorizer fit_transform..."
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(list)       
    return (X, Y)
    
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


def classification(business_file, review_file, review_count=35, binary_classify = True, cities = []): 
    (X_input, Y_input) = parseReviews(business_file, review_file, review_count, binary_classify, cities)   
    X_input = SelectKBest(chi2, k=25).fit_transform(X_input, Y_input)
    X_train, X_test, Y_train, Y_test = train_test_split(X_input, Y_input, test_size=0.4, random_state=42)


    if binary_classify == True:
        logistic_regression = LogisticRegression(C=0.1,solver='lbfgs')        
    else:
        logistic_regression = LogisticRegression(C=.1, multi_class='multinomial', solver='lbfgs')        
    Classifiers = [
        logistic_regression,
        MLPClassifier(solver='lbfgs'),
        DecisionTreeClassifier(),
        RandomForestClassifier(), 
        AdaBoostClassifier()]


    Accuracy=[]
    Model=[]    
    for classifier in Classifiers:
        try:
            a = datetime.datetime.now()
            fit = classifier.fit(X_train, Y_train)
            pred = fit.predict(X_test)
            b = datetime.datetime.now()       
            accuracy = accuracy_score(pred, Y_test)
            Accuracy.append(accuracy)
            Model.append(classifier.__class__.__name__)
            print('Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))
            print('Elapsed time = '+ str(b-a))
        except:
            pass
    if binary_classify == True:
        model = "Binary Model"
    else:
        model = "Multiclass Model"
    compBarChart(model, 'Accuracy', 'Accuracies of Models', Model, Accuracy, 45)




def main(argv):
    parser = argparse.ArgumentParser(description='Classification for yelp restaurants.')
    cities = ['Charlotte', 'Phoenix', 'Charlotte','Pittsburgh', 'Madison', 'Las Vegas'] #US
    #cities = ['Waterloo', 'Montreal'] #Canda
    #cities = ['Karlsruhe'] #Germany
    #cities = ['Edinburgh'] #UK
    classification("yelp_academic_dataset_business.json", "yelp_academic_dataset_review.json", 35, True, cities)   


if __name__ == '__main__':
    main(sys.argv)