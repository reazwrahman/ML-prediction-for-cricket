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


if PLAYER_ROLE == "BOWLER":
    data_util = BowlingDataUtil()  
    features = [
        #"date",
        "opposition",
        "ground",
        "country",
        "career_wickets_per_game",
        #"recent_wickets_per_game",
        #"career_strike_rate",
        #"recent_strike_rate",
    ]
else:
    data_util = BattingDataUtil() 
    features = [
        #"opposition",
        #"ground",
        #"country",
        "avg_runs",
        "recent_avg",
        "avg_sr",
        "recent_avg_sr",
    ] 
  
x_train = data_util.training_df

plt.figure(figsize=(16, 12))
    
for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    sns.kdeplot(x_train[x_train['bucket'] == 0][feature], label=f'Low Performance (Class 0)', fill=True, color='green')
    sns.kdeplot(x_train[x_train['bucket'] == 1][feature], label=f'High Performance (Class 1)', fill=True, color='red')
    plt.title(f'{GAME_FORMAT} {PLAYER_ROLE} Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.show()