'''
	Classifiers
	Here the code will be used to define the classifier
'''
import numpy as np
import cv2
import tensorflow as tf
import os
import sklearn
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib

# Apperance, Motion, Joint features
# cfTup = [a,b,c]	# For single sample
#
# cfTup = [[],[],[]]	# For multiple samples

def estimateClassifierValues(cfTup):
	
	yield [weightedAlpha,weightedScore]

def train_svm_classifer(features, labels, model_output_path):
    """
    train_svm_classifer will train a SVM, saved the trained and SVM model and
    report the classification performance
 
    features: array of input features
    labels: array of labels associated with the input features
    model_output_path: path for storing the trained svm model
    """
    # save 20% of data for performance evaluation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2)
 
    param = [
        {
            "kernel": ["linear"],
            "C": [1, 10, 100, 1000]
        },
        {
            "kernel": ["rbf"],
            "C": [1, 10, 100, 1000],
            "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
        }
    ]
 
    # request probability estimation
    svm = SVC(probability=True)
 
    # 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
    clf = grid_search.GridSearchCV(svm, param,
            cv=10, n_jobs=4, verbose=3)
 
    clf.fit(X_train, y_train)
 
    if os.path.exists(model_output_path):
        joblib.dump(clf.best_estimator_, model_output_path)
    else:
        print("Cannot save trained svm model to {0}.".format(model_output_path))
 
    print("\nBest parameters set:")
    print(clf.best_params_)
 
    y_predict=clf.predict(X_test)
 
    # labels=sorted(list(set(labels)))
    labels = [0,1]
    print("\nConfusion matrix:")
    print("Labels: {0}\n".format(",".join(labels)))
    print(confusion_matrix(y_test, y_predict, labels=labels))
 
    print("\nClassification report:")
    print(classification_report(y_test, y_predict))
