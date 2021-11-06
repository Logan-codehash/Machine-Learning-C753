#Logan Burke project
#September 7th 2021
#poi_id final project

import copy
import pickle
from pprint import pprint as pp

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings,
### each of which is a feature name.
### The first feature must be "poi".
features_list = [
	"poi",
    "bonus", 
    "other",
	"salary", 	
	"expenses", 
	"to_messages",
    "loan_advances", 
    "director_fees", 
	"from_messages",
	"total_payments", 
    "deferred_income",
	"fraction_to_poi",
	"restricted_stock",
    "fraction_from_poi", 
	"total_stock_value", 
    "deferral_payments",
    "expenses_per_salary", 
	"long_term_incentive", 
    "shared_receipt_with_poi",
    "exercised_stock_options",
    "from_this_person_to_poi",
    "from_poi_to_this_person",
	"restricted_stock_deferred"
	]
	
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
	data_dict = pickle.load(data_file)
 
raw_data = copy.deepcopy(data_dict)

### Explore features of the data

options = [
    "poi", 
    "people", 
    "features",  
    "nan_counts",
    "negative", 
    "non_poi"
    ] 

# pass the dataset and print out what is going on with it
def show_me_info(dataset, what):
# option 0
	if what == "poi":
		count_poi = 0
		who_are_poi = []
		for individual in dataset:
			if dataset[individual]["poi"] == True: 
				count_poi += 1
				who_are_poi.append(individual)
		print('How many poi?: %s' %(count_poi))
		print('and who are they?:')
		pp(who_are_poi)
# option 1
	elif what == "people":
		print('The number of total people are: %s' %(len(dataset)))
# option 2
	elif what == "features":
		print( 'The number of features in the dataset: %s' % (
len(dataset["METTS MARK"])))
# option 3
	elif what == "nan_counts":
		num_nans = dict((key, 0) for key, value in dataset[
            "METTS MARK"].items())
		for individual in dataset:
			for key, value in dataset[individual].items():
				if value == "NaN":
					num_nans[key] += 1
		print('Number of NaNs in dataset: ')
		pp(num_nans)
# option 4
	elif what == "negative":
		num_negs = dict((key, 0) for key, value in dataset[
            "METTS MARK"].items())
		for individual in dataset:
			for key, value in dataset[individual].items():
				if isinstance(value, int) and (value < 0):
					num_negs[key] += 1
		print('Number of negative value entries in dataset: ')
		pp(num_negs) 
# option 5
	elif what == "non_poi":
		count_non_poi = 0
		for key in dataset:
			if dataset[key]["poi"] == 0:
				count_non_poi += 1
		print('Numer of non-poi entries in the dataset: %s' %(count_non_poi))
	else:
		print('no correct option was selected. Options are:')
		pp(options)

### Task 2: Remove outliers and bad data

def nan_to_zero(dataset):
#set all NaN values in financial features to zeros
	fin_features = [ 
        "bonus", 
        "other",
        "salary", 
        "expenses", 
        "director_fees",
        "loan_advances",
        "total_payments",
        "deferred_income",
        "restricted_stock",
        "deferral_payments", 
        "total_stock_value",
        "long_term_incentive", 
        "exercised_stock_options", 
        "restricted_stock_deferred"
        ]
	for item_a in fin_features:
		for item_b in dataset:
			if dataset[item_b][item_a] == "NaN":
				dataset[item_b][item_a] = 0

#add items to remove from the data that were not good data
#Eugene was all NaN
#Total was not helpful
#the travel agency was 90% NaN and not helpful
to_remove_from_data = [
    "TOTAL",
    "LOCKHART EUGENE E",
    "THE TRAVEL AGENCY IN THE PARK"
    ]

#to remove bad data
def removing_bad_data(dataset, item):
	for removing in item:
		dataset.pop(removing, None)

#running the dataset through the first cleaner 
nan_to_zero(raw_data)

#running dataset through the second cleaner
removing_bad_data(raw_data, to_remove_from_data)

#for quicker test of options, remove for final
def test(x):
	show_me_info(raw_data, options[x])

### Task 3: Create new feature(s)
#calculates fraction of messages to/from poi
def fraction_calculation(poi_messages, all_messages):
	frac = float(0)
	if all_messages == 0:
		return 0
	if all_messages != "NaN" and poi_messages != "NaN":
		frac = float(poi_messages)/all_messages
	return frac
#creating features to test
def create_features(dataset):
	for item in dataset:
		each = dataset[item]
		from_poi = each["from_poi_to_this_person"]
		to_mess = each["to_messages"]
		fraction_from = fraction_calculation(from_poi, to_mess)
		dataset[item]["fraction_from_poi"] = fraction_from
		from_person = each["from_this_person_to_poi"]
		from_mess = each["from_messages"]
		fraction_to = fraction_calculation(from_person, from_mess)
		dataset[item]["fraction_to_poi"] = fraction_to
		expenses = dataset[item]["expenses"]
		salary = dataset[item]["salary"]
		dataset[item]["expenses_per_salary"] = \
		fraction_calculation(expenses, salary)

#run the feature creation on the copied dataset
create_features(raw_data)

### Task 4: Try a variety of classifiers
### Extract features and labels from dataset for local testing
data = featureFormat(raw_data, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#importing classifiers to test
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#classifiers to test
clf_testing = [
	GaussianNB(),
	KNeighborsClassifier(),
	DecisionTreeClassifier()
	]

#testing the classifiers
def test_clf(clf_test, data, features):
	print('This classifier is:')
	test_classifier(clf_test, data, features) 
	#test_classifier is from the tester.py file that is used for grading
	
#uncomment to run these tests
# test_clf(clf_testing[0],raw_data,features_list)
# test_clf(clf_testing[1],raw_data,features_list)
# test_clf(clf_testing[2],raw_data,features_list)


#importing tools to check which features could be important
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#checking which features are more important
#is easier to read if reverse=true
def important_features():
	pp(sorted(zip(list(mutual_info_classif(features, labels)),
	features_list[1:]), reverse = True))

#seems that decision tree gets best results most consistently
final_clf = DecisionTreeClassifier(criterion = "entropy")
testing_pipe = Pipeline(steps=[("skb",SelectKBest(mutual_info_classif)),
	("clf", final_clf)])
	
#trying to find number of min samples and number of features to use
testing_par = {"skb__k" : (3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),
	"clf__min_samples_split" : (2,3,4,5,6,7,8,9,10,11,12,13,14,15)}

#trying to maximize f1
the_grid = GridSearchCV(testing_pipe, testing_par, scoring = "f1",
	cv = 10, n_jobs = -1)

#checking fit
the_grid.fit(features, labels)

print('parameters for max F1 results:')
pp(the_grid.best_params_)
print('\n')
important_features()

#having features with their scores for manual testing
scored_features = sorted(zip(list(the_grid.best_estimator_.named_steps \
	["skb"].scores_), features_list[1:]), reverse = True)

#creating a testing feature list with the scores
highest_features = ['poi']
print("These were used in the test:")
for item in range(the_grid.best_params_['skb__k']):
  highest_features.append(scored_features[item][1])
  # displaying features for checking
  print(scored_features[item][1]," - ", scored_features[item][0])

#checking with maximized parameters
test_clf(the_grid.best_estimator_.named_steps["clf"], raw_data, \
	highest_features)

#testing final clf with most consistent parameters
#features that most often return highest results
final_features = [
	"poi",
    "bonus", 
    "other",
	"expenses", 
	"fraction_to_poi",
    "expenses_per_salary", 
    "shared_receipt_with_poi"
	]

testing_features =  [
	"poi",
    "bonus", 
    "other",
	"expenses", 
    "shared_receipt_with_poi"
	]

#clf with min sample split that most often shows up
clf = DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 6)

#which features will be used
print("Features used:")
pp(final_features[1:])
test_clf(clf, raw_data, final_features)

#dumping my data
dump_classifier_and_data(clf, raw_data, final_features)