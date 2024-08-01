import copy
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

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


class LogisticRegressionClassifier:
    def __init__(self):
        self.name = "LOGISTIC REGRESSION"
        self.general_util = Util()
        if PLAYER_ROLE == "BOWLER":
            self.data_util = BowlingDataUtil()
        else:
            self.data_util = BattingDataUtil()

        self.all_features = FEATURES

        if USE_SYNTHETIC_DATA:
            self.x_train = self.general_util.resample_data_with_smote(
                self.data_util.get_training_data(), self.all_features
            )
        else:
            self.x_train = self.data_util.get_training_data()

        self.x_test = self.data_util.get_testing_data()

        self.scaler = StandardScaler()
        self.model = None

    def update_features(self, features):
        self.all_features = features

    def scale_training_data(self):
        self.scaler.fit_transform(self.x_train[self.all_features])

    def build_model(self, training_data):
        model = LogisticRegression(random_state=42, max_iter=10000)
        model.fit(training_data, self.x_train["bucket"])
        return model

    def make_predictions(self):
        if not self.model:
            model = self.build_model(self.x_train[self.all_features])
            self.model = model
        else:
            model = self.model

        predictions = model.predict(self.x_test[self.all_features])
        return predictions

    def compute_accuracy(self, predictions):
        accuracy = accuracy_score(self.x_test["bucket"], predictions)
        return accuracy * 100

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
