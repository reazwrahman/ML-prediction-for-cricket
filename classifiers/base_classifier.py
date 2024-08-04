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

""" 
Base class (blueprint) for all classifers
"""


class BaseClassifier:
    def __init__(self):
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

        self.scale_training_data()
        self.model = None

    def scale_training_data(self):
        self.scaler = StandardScaler()
        self.scaler.fit_transform(self.x_train[self.all_features])

    def update_features(self, features):
        raise NotImplementedError

    def build_model(self, training_data):
        raise NotImplementedError

    def make_predictions(self):
        raise NotImplementedError

    def make_single_prediction(self, features_data: list):
        raise NotImplementedError

    def generate_confusion_matrix(self, predictions):
        raise NotImplementedError

    def print_confusion_matrix(self, confusion_matrix: dict):
        raise NotImplementedError
