
                  # "SKILLING JEFFREY K"

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score, precision_score

sys.path.append("../outliers/")
from outlier_cleaner import outlierCleaner

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','from_poi_to_this_person','from_this_person_to_poi'
                 ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
# We removed the values of
data_dict.pop("TOTAL")
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

### Task 3: Create new feature(s)

poi_d={}
for key,val in data_dict.items():
    if data_dict[key]['poi']==True:
        poi_d.update({key:data_dict[key]['poi']})

# to person X from some one else dictionay for future reference
to_messages={}
for key,val in data_dict.items():
    if data_dict[key]['to_messages']!='NaN':
        to_messages.update({key:data_dict[key]['to_messages']})

#from_poi_this_person dictionaty for future reference
from_poi_d={}
for key,val in data_dict.items():
    if data_dict[key]['from_poi_to_this_person']!='NaN':
        from_poi_d.update({key:data_dict[key]['from_poi_to_this_person']})

#from_poi_this_person to poi dict for future reference
from_this_2poi = {}
for key,val in data_dict.items():
    if data_dict[key]['from_this_person_to_poi']!= 'NaN':
        from_this_2poi.update({key:data_dict[key]['from_this_person_to_poi']})

from_messages = {}
for key,val in data_dict.items():
    if data_dict[key]['from_messages']!= 'NaN':
        from_messages.update({key:data_dict[key]['from_messages']})

#building a new feature here.
#This feature is to keep track of total email conversations a person has with
#a poi. I added up all the to and from conversations that a person has with a poi
# and divide them by the sum of all the to and from emails altogether for this person

for name, features in data_dict.items():
    # name = names[name]
    if features['from_this_person_to_poi']!='NaN' and features['from_messages']!='NaN':
        features['messages_to_poi_percent'] = float(features['from_this_person_to_poi'])/float(features['from_messages'])
    else:
        features['messages_to_poi_percent'] = 'NaN'

for name, features in data_dict.items():
    if features['from_poi_to_this_person']!='NaN' and features['to_messages']!='NaN':
        features['messages_from_poi_percent'] = float(features['from_poi_to_this_person'])/float(features['to_messages'])
    else:
        features['messages_to_poi_percent']='NaN'


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# print data
# print data_dict["SKILLING JEFFREY K"]
print labels
# SCALE THE DATA
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
features=scale.fit_transform(features)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
kclf = SelectKBest(chi2, k=2)
features = kclf.fit_transform(features, labels)
print 'K-best scores:,', kclf.scores_, '\n'
print '\n\n KCLF: ',kclf.get_support()


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.25, random_state=42)

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred_NB = clf.predict(features_test)
print "The accuracy score of Gaussian NB is: ", accuracy_score(labels_test,pred_NB)
print "The precision score for GaussianNB is: ", precision_score(labels_test, pred_NB, average='weighted')
# #The accuracy score of Gaussian NB is:  0.909090909091 and precision score is 0.826446280992

#Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(features_train, labels_train)
pred_tree = clf.predict(features_test, labels_test)
print "\n\ntest labels for DT: ", len(labels_test), "predicted values for DT:", len(pred_tree)
print "The accuracy of decision tree is :", accuracy_score(labels_test, pred_tree)
print "The precision for decision tree is :", precision_score(labels_test, pred_tree, average='weighted')
# #The accuracy score for Decision Tree Classifer is 0.878787878788 and presicion score is  0.823863636364


#Support Vector Machine(SVM)
from sklearn.svm import SVC
clf= SVC()
clf.fit(features_train, labels_train)
pred_svc = clf.predict(features_test)
print "\n\ntest labels for SVM: ", len(labels_test), "predicted values for SVM:", len(pred_svc)
print "The accuracy of SVM is :", accuracy_score(labels_test, pred_svc)
print "The precision for SVM is :", precision_score(labels_test, pred_svc, average='weighted')
# #The accuracy score for SVM is 0.909090909091 and presicion score is  0.826446280992

#Random Forrest Regressor
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, max_depth=1, random_state=42)
clf.fit(features_train, labels_train)
for_pred = clf.predict(features_test)
print "\n\ntest labels for Forrest: ", len(labels_test), "predicted values for Forrest:", len(for_pred)
print "The accuracy of RF is :", accuracy_score(labels_test, for_pred)
print "The precision for RF is :", precision_score(labels_test, for_pred, average='weighted')
# #The accuracy score for RF is 0.848484848485 and presicion score is  0.869122257053


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
