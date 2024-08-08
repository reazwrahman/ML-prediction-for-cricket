import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import copy
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

## initialize (if known) for faster computation
OPTIMAL_K = 16  ## set None if not known


class KNNClassifier(BaseClassifier):
    def __init__(self):
        self.name = "KNN"
        super().__init__()
        if OPTIMAL_K:
            self.optimal_k = OPTIMAL_K
        else:
            self.optimal_k = self.find_optimal_k()

    def build_model(self, training_data, k=None):
        if k is None:
            k = self.optimal_k
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(training_data, self.x_train["bucket"])
        self.feature_importance = permutation_importance(
            model,
            training_data,
            self.x_train["bucket"],
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        return model

    ## override base method
    def make_predictions(self, k=None):
        if not k:
            k = self.optimal_k

        if not self.model:
            self.model = self.build_model(self.x_train[self.all_features], k)

        predictions = self.model.predict(self.x_test[self.all_features])
        self.x_test["predictions"] = predictions
        return predictions

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

    def get_feature_importance(self):
        feature_importances_df = pd.DataFrame(
            {
                "Feature": self.all_features,
                "Weight": self.feature_importance.importances_mean,
            }
        )
        self.feature_importance = feature_importances_df.sort_values(
            by="Weight", ascending=False
        ).reset_index(drop=True)

        return self.feature_importance


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
    print(f"optimal k = {classifier.optimal_k}")
    print("\n")

    classifier.build_model(classifier.x_train[classifier.all_features])
    imp = classifier.get_feature_importance()
    print(imp)
