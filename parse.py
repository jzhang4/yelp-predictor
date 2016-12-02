
"""
Extracts features for restaurant businesses in the Yelp Challenge Dataset
"""

import sys
from json import loads
from json import dumps
from re import sub


"""
Returns true if a file ends in .json
"""
def isJson(f):
    return len(f) > 5 and f[-5:] == '.json'


"""
Parses a single json file. Currently, there's a loop that iterates over each
item in the data set. Your job is to extend this functionality to create all
of the necessary SQL tables for your database.
"""
def parseJson(json_file):
    aggregated_features = set()
    businesses = []
    with open(json_file, 'r') as f:
        for line in f:
            businesses.append(loads(line))

    data = []
    success_count = 0
    for business in businesses:
        if ('Restaurants' in business['categories']) and (business['review_count'] > 35) and (business['city'] == 'Las Vegas'):
            success = 1 if business['stars'] > 3.5 else -1
            if success == 1:
                success_count += 1
            attributes = business['attributes']
            features = {}
            for key in business['categories']:
                if key != 'Restaurants':
                    features[key] = 1
            for neighborhood in business['neighborhoods']:
                features[key] = 1
            for key in attributes:
                if key == 'Price Range':
                    feature_key = "Price Range " + str(attributes[key])
                    features[feature_key] = 1
                elif key == 'Attire':
                    feature_key = "Attire " + str(attributes[key])
                    features[feature_key] = 1
                elif key == 'Alcohol':
                    feature_key = "Alcohol " + str(attributes[key])
                    features[feature_key] = 1
                elif key == 'Noise Level':
                    feature_key = "Noise Level " + str(attributes[key])
                    features[feature_key] = 1
                elif key == 'Wifi':
                    feature_key = "Wifi " + str(attributes[key])
                    features[feature_key] = 1  
                elif key == 'Smoking':
                    feature_key = "Smoking " + str(attributes[key])
                    features[feature_key] = 1 
                elif key == 'Ages Allowed':
                    feature_key = "Ages Allowed  " + str(attributes[key])
                    features[feature_key] = 1
                elif key == 'BYOB/Corkage':
                    feature_key = "BYOB/Corkage " + str(attributes[key])
                    features[feature_key] = 1       
                elif key == 'Parking':
                    parking_options = attributes[key]
                    for option in parking_options:
                        if parking_options[option] == True:
                            features[("Parking " + option)] = 1
                elif key == 'Good For':
                    good_for = attributes[key]
                    for option in good_for:
                        if good_for[option] == True:
                            features[("Good For " + option)] = 1
                elif key == 'Ambience':
                    ambiences = attributes[key]
                    for ambience in ambiences:
                        if ambiences[ambience] == True:
                            features[("Ambience " + ambience)] = 1
                else:
                    if attributes[key] == True:
                        features[key] = 1
            for key in features:
                aggregated_features.add(key)
            data.append((features, success))

    print("Successful Restaurants: " + str(success_count))
    print("Total Restaurants: " + str(len(data)))

    data_file = open('restaurant_data.json','w')
    for business in data:
        data_file.write(dumps(business, data_file))
        data_file.write('\n')
    data_file.close()

    aggregated_features_file = open('aggregated_features.txt','w')
    for feature in aggregated_features:
        aggregated_features_file.write(feature)
        aggregated_features_file.write('\n')
    aggregated_features_file.close()

"""
Loops through each json files provided on the command line and passes each file
to the parser
"""
def main(argv):
    f = "yelp_academic_dataset_business.json"
    if isJson(f):
        parseJson(f)
        print "Success parsing " + f

if __name__ == '__main__':
    main(sys.argv)
