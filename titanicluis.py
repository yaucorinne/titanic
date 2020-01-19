#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:59:12 2019

@author: luischavesrodriguez
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import math
from sklearn.impute import KNNImputer
from impyute.imputation.cs import mice
import datawig
import xgboost as xgb
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors, metrics, svm, ensemble

# original datasets, do not modify
trainO = pd.read_csv('data/train.csv')
testO = pd.read_csv('data/test.csv')

# remove rows with missing data for embarked
train = trainO[~trainO['Embarked'].isna()]
test = testO[~testO['Embarked'].isna()]

# take first character of cabin (letter) assuming that gives us further info,
# otherwise all cabin numbers are different and don't give us any info
train['Cabin'][~train['Cabin'].isna()] = [c[0] for c in train['Cabin'] if isinstance(c, str)]
test['Cabin'][~test['Cabin'].isna()] = [c[0] for c in test['Cabin'] if isinstance(c, str)]

#store passengerID for future output
testPID = test['PassengerId']

# remove name, passenger id, ticket id as not useful variable for analysis
train = train.loc[:, ~train.columns.isin(['PassengerId', 'Name', 'Ticket'])]
test = test.loc[:, ~test.columns.isin(['PassengerId', 'Name', 'Ticket'])]

# take variables for which you want to do one hot encoding
toEncode = train.loc[:, ['Sex', 'Embarked']]
enc = OneHotEncoder(handle_unknown='ignore')
HotCategorical = enc.fit_transform(toEncode).toarray()
train = np.concatenate((train.loc[:, ~train.columns.isin(['Sex', 'Embarked'])],
                        HotCategorical), axis=1)
## same for test
HotCategorical_test = enc.transform(toEncode_test).toarray()
test = np.concatenate((test.loc[:, ~test.columns.isin(['Sex', 'Embarked'])],
                       HotCategorical_test), axis=1)

# removing cabin for now
mask_train = np.ones(train.shape[1], dtype=bool)
mask_train[6] = False
trainPreImpute = train[:, mask_train]
## same for test
mask_test = np.ones(test.shape[1], dtype=bool)
mask_test[5] = False
testPreImpute = test[:, mask_test]

# data imputation
imputer = KNNImputer(n_neighbors=2, weights="uniform")
trainImp = imputer.fit_transform(trainPreImpute)
# train = np.concatenate((trainImp, train[:, ~mask]), axis=1)
train = trainImp

## Same for test
# data imputation
imputer = KNNImputer(n_neighbors=2, weights="uniform")
testImp = imputer.fit_transform(testPreImpute)
# train = np.concatenate((trainImp, train[:, ~mask]), axis=1)
test = testImp

y_train = train[:, 0]
X_train = train[:, 1:]
X_test = test


# Testing different models

## KNN
for n in range(1, 21):
    nbrs = neighbors.KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    score = cross_val_score(nbrs, X_train, y_train, cv=5, scoring='accuracy')
    f1 = cross_val_score(nbrs, X_train, y_train, cv=5, scoring='f1')
    print("%d neighbour(s) -> Accuracy: %0.2f (+/- %0.2f)\t F1-score: %0.2f (+/-%0.2f)"
          % (n, score.mean(), score.std() * 2, f1.mean(), f1.std() * 2))

# NAIVE BAYES
gnb = GaussianNB()
score = cross_val_score(gnb, X_train, y_train, cv=5, scoring='accuracy')
recall = cross_val_score(gnb, X_train, y_train, cv=5, scoring='recall')
f1 = cross_val_score(gnb, X_train, y_train, cv=5, scoring='f1')
print("Naive-Bayes classifier -> Accuracy: %0.2f (+/- %0.2f)\t Recall: %0.2f (+/- %0.2f)\t F1-score: %0.2f (+/- %0.2f)"
      % (score.mean(), score.std() * 2, recall.mean(), recall.std() * 2, f1.mean(), f1.std() * 2))

# SVM
for k in ['linear', 'poly']:
    if k == 'poly':
        for d in range(2):
            SVM = svm.SVC(random_state=0, tol=1e-1, max_iter=1000000, kernel=k, degree=d, gamma='scale')
            score = cross_val_score(SVM, X_train, y_train, cv=5, scoring='accuracy')
            recall = cross_val_score(SVM, X_train, y_train, cv=5, scoring='recall')
            print("Kernel ", k, ", polynomial degree ", d, "\n\t Acc:", round(score.mean(), 2), "(+/-)",
                  round(score.std() * 2, 2), "\t ", "Recall:",
                  round(recall.mean(), 2), "(+/-)",
                  round(recall.std() * 2, 2))
    else:
        SVM = svm.SVC(random_state=0, tol=1e-1, max_iter=1000000, kernel=k, gamma='scale')
        score = cross_val_score(SVM, X_train, y_train, cv=5, scoring='accuracy')
        recall = cross_val_score(SVM, X_train, y_train, cv=5, scoring='recall')
        print("Kernel ", k, "\n\t Acc:", round(score.mean(), 2), "(+/ -)", round(score.std() * 2, 2), "\t ",
              "Recall:", round(recall.mean(), 2), "(+/-)", round(recall.std() * 2, 2))

# Random forest sklearn
for n in range(5, 50, 5):
    RFC = ensemble.RandomForestClassifier(max_depth=n)
    score = cross_val_score(RFC, X_train, y_train, cv=5, scoring='accuracy')
    recall = cross_val_score(RFC, X_train, y_train, cv=5, scoring='recall')
    f1 = cross_val_score(RFC, X_train, y_train, cv=5, scoring='f1')
    print("Random Forest with depth %d -> Accuracy: %0.2f (+/- %0.2f)\t "
          "F1-score: %0.2f (+/- %0.2f)"
          % (n, score.mean(), score.std() * 2, f1.mean(), f1.std() * 2))

# XGboost -- specify parameters via map
boosty = xgb.XGBClassifier()
acc = cross_val_score(boosty, X_train, y_train, cv = 5, scoring = 'accuracy')
print("XGBoost -> Accuracy: %0.2f (+/- %0.2f)\t "
          % (acc.mean(), acc.std() * 2))


# final model
#FM = ensemble.RandomForestClassifier(max_depth=10)

FM = boosty
FM.fit(X_train, y_train)
predictions = FM.predict(X_test)

# predictions to submittable folder
predictionFile = pd.DataFrame(np.stack([testPID, predictions]).T,
                              columns = ['PassengerId','Survived'])
predictionFile = predictionFile.astype(int)
predictionFile.to_csv("predictions3.csv", index = False)

# need to do all things I've done to train to test, get results and submit
# also, try with nan as categorical values and hot encoded

# imputer = SimpleImputer(
#     input_columns=['1', '3', '4', '5', '7', '8', '9', '10', '11'],
#     # column(s) containing information about the column we want to impute
#     output_column='2',  # the column we'd like to impute values for
#     output_path='imputer_model'  # stores model data and metrics
# )
#
# imputer2 = SimpleImputer(
#     input_columns=['1', '2','3', '4', '5', '7', '8', '9', '10', '11'],
#     # column(s) containing information about the column we want to impute
#     output_column='6',  # the column we'd like to impute values for
#     output_path='imputer_model2'  # stores model data and metrics
# )

# imputedTrain = datawig.SimpleImputer.complete(train2)
