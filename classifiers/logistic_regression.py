import copy
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self):
        self.name = "LOGISTIC REGRESSION"
        super().__init__()

    def build_model(self, training_data):
        model = LogisticRegression(random_state=42, max_iter=10000)
        model.fit(training_data, self.x_train["bucket"])
        return model

    def experiment_dropping_feature(self):
        for i in range(len(self.all_features)):
            selected_features = copy.deepcopy(self.all_features)
            print(f"logistic regression::dropping feature {self.all_features[i]}:")
            del selected_features[i]

            model = self.build_model(self.x_train[selected_features])
            predictions = model.predict(self.x_test[selected_features])
            accuracy = classifier.compute_accuracy(predictions)
            print(accuracy)
            print("\n")


if __name__ == "__main__":
    print(
        f"GAME FORMAT: {GAME_FORMAT}, PREDICTION FORMAT: {PREDICTION_FORMAT}, PLAYER ROLE: {PLAYER_ROLE}"
    )
    if PREDICTION_FORMAT == "BINARY":
        classifier = LogisticRegressionClassifier()
        predictions = classifier.make_predictions()
        accuracy = classifier.compute_accuracy(predictions)
        print(f"logistic regression all features used")
        print(accuracy)
        print("\n")
        # classifier.experiment_dropping_feature()
        classifier.print_confusion_matrix(
            classifier.generate_confusion_matrix(predictions)
        )
        print(classifier.x_test["predictions"].unique())
    else:
        print("Logistic Regression can only be applied for binary predictions")
