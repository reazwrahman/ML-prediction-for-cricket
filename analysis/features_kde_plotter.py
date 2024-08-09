import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

print(
    f"GAME FORMAT: {GAME_FORMAT}, PREDICTION FORMAT: {PREDICTION_FORMAT}, PLAYER ROLE: {PLAYER_ROLE}"
)

if PLAYER_ROLE == "BOWLER":
    data_util = BowlingDataUtil()
else:
    data_util = BattingDataUtil()

x_train = data_util.training_df

features_set = [FEATURES[0:4], FEATURES[4:8], FEATURES[8::]]

for features in features_set:
    plt.figure(figsize=(16, 12))

    for i, feature in enumerate(features):
        plt.subplot(2, 2, i + 1)
        sns.kdeplot(
            x_train[x_train["bucket"] == 0][feature],
            label=f"Low Performance (Class 0)",
            fill=True,
            color="green",
        )
        sns.kdeplot(
            x_train[x_train["bucket"] == 1][feature],
            label=f"High Performance (Class 1)",
            fill=True,
            color="red",
        )
        plt.title(f"feature: {feature}")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.legend()

    plt.tight_layout()
    plt.show()
