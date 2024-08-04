import copy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
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

## NOTE: initialize with values if known (for faster runtime)
BEST_N = 200
BEST_D = 10


class MyRandomForestClassifier(BaseClassifier):
    def __init__(self):
        self.name = "RANDOM FOREST"
        super().__init__()

        self.best_n_estimators = BEST_N
        self.best_max_depth = BEST_D  

        self.feature_weights = None 
        self.feature_importance = None

    def update_features(self, features):
        self.all_features = features

    def __find_optimal_parameters(self, training_data):
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": [10, 20, 30, 50, 100, 200],
            "max_depth": [5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        # Use GridSearchCV to find the best parameters
        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1
        )
        grid_search.fit(training_data, self.x_train["bucket"])

        # Use the best estimator found by GridSearchCV
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        self.best_n_estimators = best_params["n_estimators"]
        self.best_max_depth = best_params["max_depth"]
        return self.best_n_estimators, self.best_max_depth

    def build_model(self, training_data):
        if not self.best_n_estimators or not self.best_max_depth:
            self.__find_optimal_parameters(training_data)

        model = RandomForestClassifier(
            n_estimators=self.best_n_estimators,
            max_depth=self.best_max_depth,
            criterion="entropy",
            random_state=42,
        )
        model.fit(training_data, self.x_train["bucket"])
        return model

    def make_predictions(self):
        if not self.model:
            model = self.build_model(self.x_train[self.all_features])
            self.model = model
        else:
            model = self.model

        predictions = model.predict(self.x_test[self.all_features]) 
        self.feature_weights = model.feature_importances_
        return predictions

    def compute_accuracy(self, predictions):
        accuracy = accuracy_score(self.x_test["bucket"], predictions)
        return accuracy * 100

    def make_single_prediction(self, features_data: list):
        if not self.model:
            model = self.build_model(self.x_train[self.all_features])
            self.model = model
        else:
            model = self.model

        prediction = model.predict([features_data])
        return prediction

    def generate_confusion_matrix(self, predictions):
        return self.general_util.generate_confusion_matrix(predictions, self.x_test)

    def print_confusion_matrix(self, confusion_matrix: dict):
        self.general_util.print_confusion_matrix(confusion_matrix)

    def get_optimal_parameters(self):
        return (self.best_n_estimators, self.best_max_depth) 
    
    def get_feature_importance(self): 
        feature_importances_df = pd.DataFrame({
            'Feature': self.all_features,
            'Weight': self.feature_weights
        })

        self.feature_importance = feature_importances_df.sort_values(by='Weight', ascending=False).reset_index(drop=True)
        return self.feature_importance

if __name__ == "__main__":
    print(
        f"GAME FORMAT: {GAME_FORMAT}, PREDICTION FORMAT: {PREDICTION_FORMAT}, PLAYER ROLE: {PLAYER_ROLE}"
    )
    classifier = MyRandomForestClassifier()
    predictions = classifier.make_predictions()
    accuracy = classifier.compute_accuracy(predictions)
    print(f"{classifier.name} all features used")
    print(accuracy)
    print("\n")
    # classifier.experiment_dropping_feature()
    classifier.print_confusion_matrix(classifier.generate_confusion_matrix(predictions))
    print(
        f"optimal n = {classifier.get_optimal_parameters()[0]}, optimal depth = {classifier.get_optimal_parameters()[1]}"
    )  
    print('\n')
    print(classifier.get_feature_importance())
