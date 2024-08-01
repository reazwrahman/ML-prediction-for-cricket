GAME_FORMAT = "ODI"  ## ODI/T20

PREDICTION_FORMAT = "BINARY"  ## BINARY/CUSTOM

PLAYER_ROLE = "BOWLER"  ## BATTER/BOWLER

USE_SYNTHETIC_DATA = True ## to address the imbalance of minority labels

if PLAYER_ROLE == "BOWLER":
    FEATURES = [
        "opposition",
        "ground",
        "country",
        "career_wickets_per_game",
        "recent_wickets_per_game",
        "career_strike_rate",
        "recent_strike_rate",
    ]

else:
    FEATURES = [
        "opposition",
        "ground",
        "country",
        "avg_runs",
        "recent_avg",
        "avg_sr",
        "recent_avg_sr",
    ]  ## features used in classifiers

# source data: https://data.world/cclayford/cricinfo-statsguru-data
DATA_FILES = {
    "T20": {
        "training_files": [
            "data/t20/Men T20I Player Innings Stats - 2011 to 2020 Team Group 1.csv",
            "data/t20/Men T20I Player Innings Stats - 2011 to 2020 Team Group 2.csv",
            "data/t20/Men T20I Player Innings Stats - 2011 to 2020 Team Group 3.csv",
            "data/t20/Men T20I Player Innings Stats - 2011 to 2020 Team Group 4.csv",
            "data/t20/Men T20I Player Innings Stats - 2011 to 2020 Team Group 5.csv",
            "data/t20/Men T20I Player Innings Stats - 2021.csv",
        ],
        "testing_files": [
            "data/t20/Men T20I Player Innings Stats - 2022.csv",
            "data/t20/Men T20I Player Innings Stats - 2023.csv",
            "data/t20/Men T20I Player Innings Stats - 2024.csv",
        ],
    },
    "ODI": {
        "training_files": [
            "data/odi/Men ODI Player Innings Stats - 20th Century.csv",
            "data/odi/Men ODI Player Innings Stats - 2001 to 2010 Team Group 1.csv",
            "data/odi/Men ODI Player Innings Stats - 2001 to 2010 Team Group 2.csv",
            "data/odi/Men ODI Player Innings Stats - 2001 to 2010 Team Group 3.csv",
            "data/odi/Men ODI Player Innings Stats - 2001 to 2010 Team Group 4.csv",
            "data/odi/Men ODI Player Innings Stats - 2001 to 2010 Team Group 5.csv",
            "data/odi/Men ODI Player Innings Stats - 2011 to 2020 Team Group 1.csv",
            "data/odi/Men ODI Player Innings Stats - 2011 to 2020 Team Group 2.csv",
            "data/odi/Men ODI Player Innings Stats - 2011 to 2020 Team Group 3.csv",
        ],
        "testing_files": [
            "data/odi/Men ODI Player Innings Stats - 2011 to 2020 Team Group 4.csv",
            "data/odi/Men ODI Player Innings Stats - 2011 to 2020 Team Group 5.csv",
            "data/odi/Men ODI Player Innings Stats - 2021.csv",
            "data/odi/Men ODI Player Innings Stats - 2022.csv",
            "data/odi/Men ODI Player Innings Stats - 2023.csv",
            "data/odi/Men ODI Player Innings Stats - 2024.csv",
        ],
    },
}
