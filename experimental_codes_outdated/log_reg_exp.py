#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 15:27:18 2024

@author: reazrahman
"""

# global imports
import copy
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# local imports
from data.batting_data import BattingDataGenerator

training_filenames = [
    "data/t20/Men T20I Player Innings Stats - 2021.csv",
    "data/t20/Men T20I Player Innings Stats - 2022.csv",
]


def encode_data(x_train):
    ## encode features
    le = LabelEncoder()
    x_train["opposition"] = le.fit_transform(x_train["opposition"])
    x_train["ground"] = le.fit_transform(x_train["ground"])
    x_train["country"] = le.fit_transform(x_train["country"])
    x_train["bucket"] = le.fit_transform(x_train["bucket"])

    return x_train


def scale_data(x_train):
    ## scale the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train[selected_features])

    return x_train


all_features = [
    "opposition",
    "ground",
    "country",
    "avg_runs",
    "recent_avg",
    "avg_sr",
    "recent_avg_sr",
]

for i in range(len(all_features)):
    selected_features = copy.deepcopy(all_features)

    print(f"dropping feature {all_features[i]}:")
    del selected_features[i]

    x_train_generator = BattingDataGenerator(filenames=training_filenames)
    x_train = x_train_generator.df

    x_train = encode_data(x_train)
    x_train_labels = x_train["bucket"]
    x_train_features = x_train[selected_features]

    ## apply logistic regressor to training data
    log_reg = LogisticRegression(random_state=42, max_iter=10000)
    log_reg.fit(x_train_features, x_train_labels)

    ## test data
    test_file_names = [
        "data/t20/Men T20I Player Innings Stats - 2023.csv",
        "data/t20/Men T20I Player Innings Stats - 2024.csv",
    ]
    test_data_generator = BattingDataGenerator(filenames=test_file_names)
    x_test = test_data_generator.df

    x_test = encode_data(x_test)
    x_test_labels = x_test["bucket"]

    ## predict
    predictions = log_reg.predict(x_test[selected_features])
    accuracy = accuracy_score(x_test_labels, predictions)
    print(accuracy * 100)


selected_features = copy.deepcopy(all_features)

print(f"keeping all features")
del selected_features[i]


x_train_generator = BattingDataGenerator(filenames=training_filenames)
x_train = x_train_generator.df

x_train = encode_data(x_train)
x_train_labels = x_train["bucket"]
x_train_features = x_train[selected_features]

## apply logistic regressor to training data
log_reg = LogisticRegression(random_state=42, max_iter=10000)
log_reg.fit(x_train_features, x_train_labels)


## test data
test_file_names = [
    "data/t20/Men T20I Player Innings Stats - 2023.csv",
    "data/t20/Men T20I Player Innings Stats - 2024.csv",
]
test_data_generator = BattingDataGenerator(filenames=test_file_names)
x_test = test_data_generator.df

x_test = encode_data(x_test)
x_test_labels = x_test["bucket"]

## predict
predictions = log_reg.predict(x_test[selected_features])
accuracy = accuracy_score(x_test_labels, predictions)
print(accuracy * 100)
