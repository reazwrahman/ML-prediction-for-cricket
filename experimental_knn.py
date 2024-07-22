#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 20:57:31 2024

@author: reazrahman
"""

# global imports
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score  
import matplotlib.pyplot as plt

from batting_data import BattingDataUtil  


batting_util = BattingDataUtil()
x_train = batting_util.get_training_data() 
x_test = batting_util.get_testing_data()

features = batting_util.selected_features 

#features = ['opposition','ground', 'country', 'avg_runs', 'recent_avg', 
#                   'avg_sr', 'recent_avg_sr'] 

#features=['avg_runs', 'recent_avg']

x_train_features = x_train[features]
x_train_labels = x_train['bucket']

x_test_features = x_test[features]
x_test_labels = x_test['bucket'] 



##### Define the range of k values to test
k_values = [13, 15, 16, 18, 20, 22, 25, 30, 32, 34, 36, 38, 42, 45, 48, 50]

##### Store accuracy results
accuracy_results = []

##### Train and evaluate k-NN classifier for each k 

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train_features, x_train_labels)
    predictions = knn.predict(x_test_features)
    accuracy = accuracy_score(x_test_labels, predictions)
    accuracy_results.append(accuracy*100)
    print(f'k={k}, Accuracy={accuracy*100:.4f}')
    
print('\n') 


## question 3.2: plot accuracy 
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_results, marker='o')
plt.title('k-NN Classifier Accuracy for Different k Values')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show() 