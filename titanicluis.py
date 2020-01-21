#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:59:12 2019

@author: luischavesrodriguez
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
import xgboost as xgb
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors, metrics, svm, ensemble

# original datasets, do not modify
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# take first character of cabin (letter) assuming that gives us further info,
# otherwise all cabin numbers are different and don't give us any info
train.loc[~train['Cabin'].isna(), 'Cabin'] = [c[0] for c in train['Cabin'] if isinstance(c, str)]
test.loc[~test['Cabin'].isna(), 'Cabin'] = [c[0] for c in test['Cabin'] if isinstance(c, str)]
# because this pre-processing is only performing on missing values I am not sure it could be pipelined

# store passengerID for future output
testPID = test['PassengerId']

# remove name, passenger id, ticket id as not useful variable for analysis
train.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# Remove rows with missing target, separate target from predictors
train.dropna(axis=0, subset=['Survived'], inplace=True)
y_train = train.Survived
X_train = train.copy()
X_test = test.copy()
X_train.drop(['Survived'], axis=1, inplace=True)

# define categorical and numerical columns
# All categorical columns
categorical_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
# Numerical columns (remaning columns)
numerical_cols = list(set(X_train.columns) - set(categorical_cols))

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = [xgb.XGBClassifier(n_estimators=200),
         GaussianNB(),
         RandomForestClassifier(random_state=0)]

best_score = 0
best_model = ""
results = {}
for k, m in enumerate(model):
    # Bundle pre-processing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', m)
                                  ])

    scores = cross_val_score(my_pipeline,
                             X_train, y_train,
                             cv=5, scoring='accuracy')

    #update best model if it improves from the previous best
    best_model = m if (scores.mean() > best_score) else best_model
    best_score = scores.mean() if (scores.mean() > best_score) else best_score

    #store results in a dictionary
    results[k] = {'Model': str(m).split('(')[0],
                  'Mean': scores.mean(),
                  'SD': scores.std()}

    print("Model %s gave an accuracy of %0.2f (+/-) %0.2f"
          % (str(m).split('(')[0], scores.mean(), scores.std()))

results = pd.DataFrame(results).T

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', best_model)
                              ])

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)
# Preprocessing of validation data, get predictions
predictions = my_pipeline.predict(X_test)

# # predictions to submittable folder
predictionFile = pd.DataFrame(np.stack([testPID, predictions]).T,
                              columns=['PassengerId', 'Survived'])
predictionFile = predictionFile.astype(int)
predictionFile.to_csv("predictions4.csv", index=False)
