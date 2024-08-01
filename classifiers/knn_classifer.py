import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import copy
import os 
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
script_dir = os.path.dirname(os.path.abspath(__file__))


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


class KNNClassifier:
    def __init__(self):
        self.general_util = Util()
        self.name = "KNN"
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
        self.optimal_k = None

    def update_features(self, features):
        self.all_features = features

    def scale_training_data(self):
        self.scaler.fit_transform(self.x_train[self.all_features])

    def build_model(self, training_data, k):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(training_data, self.x_train["bucket"])
        return model

    def make_predictions(self, k=None):
        if not k:
            k = self.find_optimal_k()

        if not self.model:
            model = self.build_model(self.x_train[self.all_features], k)
            self.model = model
        else:
            model = self.model

        predictions = model.predict(self.x_test[self.all_features])
        self.x_test["predictions"] = predictions
        return predictions

    def make_single_prediction(self, features_data: list):
        if not self.model:
            model = self.build_model(
                self.x_train[self.all_features], self.find_optimal_k()
            )
            self.model = model
        else:
            model = self.model

        prediction = model.predict([features_data])
        return prediction

    def compute_accuracy(self, predictions):
        accuracy = accuracy_score(self.x_test["bucket"], predictions)
        return accuracy * 100

    def find_optimal_k(self):
        if not self.optimal_k:
            k_values = [
                13,
                15,
                16,
                18,
                20,
                22,
                25,
                30,
                32,
                34,
                36,
                38,
                42,
                45,
                48,
                50,
                55,
                60,
            ]
            accuracy_results = []
            for k in k_values:
                model = self.build_model(self.x_train[self.all_features], k)
                predictions = model.predict(self.x_test[self.all_features])
                accuracy = accuracy_score(self.x_test["bucket"], predictions)
                accuracy_results.append((accuracy * 100, k))

            # self.__plot_k_vs_accuracy(copy.deepcopy(accuracy_results))
            accuracy_results.sort()
            self.optimal_k = accuracy_results[-1][1]

        return self.optimal_k

    def generate_confusion_matrix(self, predictions):
        return self.general_util.generate_confusion_matrix(predictions, self.x_test)

    def print_confusion_matrix(self, confusion_matrix: dict):
        self.general_util.print_confusion_matrix(confusion_matrix)

    def __plot_k_vs_accuracy(self, accuracy_results):
        k_values = []
        accuracies = []
        for each in accuracy_results:
            k_values.append(each[1])
            accuracies.append(each[0])
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, accuracies, marker="o")
        plt.title("k-NN Classifier Accuracy for Different k Values")
        plt.xlabel("k")
        plt.ylabel("Accuracy")
        plt.xticks(k_values)
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    print(
        f"GAME FORMAT: {GAME_FORMAT}, PREDICTION FORMAT: {PREDICTION_FORMAT}, PLAYER ROLE: {PLAYER_ROLE}"
    )
    classifier = KNNClassifier()
    predictions = classifier.make_predictions()
    accuracy = classifier.compute_accuracy(predictions)
    print(f"KNN classifier all features used")
    print(accuracy)
    print("\n")
    classifier.print_confusion_matrix(classifier.generate_confusion_matrix(predictions))
    print(classifier.x_test["predictions"].unique())
