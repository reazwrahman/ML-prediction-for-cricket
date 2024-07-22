GAME_FORMAT = "T20" ## ODI/T20 

PREDICTION_FORMAT = "CUSTOM" # BINARY/CUSTOM

# source data: https://data.world/cclayford/cricinfo-statsguru-data 
DATA_FILES = { 
    "T20": {"training_files": ["data/t20/Men T20I Player Innings Stats - 2021.csv", "data/t20/Men T20I Player Innings Stats - 2022.csv"], 
            "testing_files": ["data/t20/Men T20I Player Innings Stats - 2023.csv", "data/t20/Men T20I Player Innings Stats - 2024.csv"] 
        }, 

    "ODI": {"training_files": ["data/odi/Men ODI Player Innings Stats - 2021.csv", "data/odi/Men ODI Player Innings Stats - 2022.csv"], 
            "testing_files": ["data/odi/Men ODI Player Innings Stats - 2023.csv", "data/odi/Men ODI Player Innings Stats - 2024.csv"] 
        }, 
}
  
