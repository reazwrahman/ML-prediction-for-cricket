import pandas as pd
from sklearn.svm import SVC
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


class SVMClassifier(BaseClassifier):
    def __init__(self):
        self.name = "SVM"
        super().__init__()

    def build_model(self, training_data):
        model = SVC(probability=True)
        model.fit(training_data, self.x_train["bucket"])
        return model


if __name__ == "__main__":
    print(
        f"GAME FORMAT: {GAME_FORMAT}, PREDICTION FORMAT: {PREDICTION_FORMAT}, PLAYER ROLE: {PLAYER_ROLE}"
    )
    classifier = SVMClassifier()
    predictions = classifier.make_predictions()
    accuracy = classifier.compute_accuracy(predictions)
    print(f"SVM classifier all features used")
    print(accuracy)
    print("\n")
    classifier.print_confusion_matrix(classifier.generate_confusion_matrix(predictions))
