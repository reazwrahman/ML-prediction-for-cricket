import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.model_selection import GridSearchCV
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
script_dir = os.path.dirname(os.path.abspath(__file__))

from classifiers.base_classifier import BaseClassifier
from data.batting_data import BattingDataUtil
from data.bowling_data import BowlingDataUtil
from config import (
    GAME_FORMAT,
    PREDICTION_FORMAT,
    FEATURES,
    PLAYER_ROLE,
    USE_SYNTHETIC_DATA,
)
from util import Util 

class GBMClassifier(BaseClassifier):
    def __init__(self):
        self.name = "GRADIENT BOOSTING"
        super().__init__() 
    
    def find_optimal_parameters(self, training_data): 

        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Instantiate the model
        model = GradientBoostingClassifier()
        
        # Use GridSearchCV to find the best parameters
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(training_data[self.all_features], training_data['bucket'])
        
        # Use the best model found by GridSearchCV
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_ 
        print (best_params)
        return best_params

    def build_model(self, training_data): 
        print('gbm build model is called')
        model = GradientBoostingClassifier() 
        model.fit(training_data, self.x_train["bucket"])
        return model

if __name__ == "__main__": 
    print(
        f"GAME FORMAT: {GAME_FORMAT}, PREDICTION FORMAT: {PREDICTION_FORMAT}, PLAYER ROLE: {PLAYER_ROLE}"
    )
    
    classifier = GBMClassifier()
    predictions = classifier.make_predictions()
    accuracy = classifier.compute_accuracy(predictions)
    print(f"GBM all features used")
    print(accuracy)
    print("\n")
    classifier.print_confusion_matrix(
        classifier.generate_confusion_matrix(predictions)
    )
    classifier.find_optimal_parameters(classifier.x_train)
   
