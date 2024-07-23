GAME_FORMAT = "T20" ## ODI/T20 

PREDICTION_FORMAT = "BINARY" # BINARY/CUSTOM

# source data: https://data.world/cclayford/cricinfo-statsguru-data 
DATA_FILES = { 
    "T20": {"training_files": [
                                "data/t20/Men T20I Player Innings Stats - 2011 to 2020 Team Group 1.csv", 
                                "data/t20/Men T20I Player Innings Stats - 2011 to 2020 Team Group 2.csv",  
                                "data/t20/Men T20I Player Innings Stats - 2011 to 2020 Team Group 3.csv",  
                                "data/t20/Men T20I Player Innings Stats - 2011 to 2020 Team Group 4.csv",  
                                "data/t20/Men T20I Player Innings Stats - 2011 to 2020 Team Group 5.csv", 
                                "data/t20/Men T20I Player Innings Stats - 2021.csv"  
                               ], 

            "testing_files": [ 
                                "data/t20/Men T20I Player Innings Stats - 2022.csv",
                                "data/t20/Men T20I Player Innings Stats - 2023.csv",  
                                "data/t20/Men T20I Player Innings Stats - 2024.csv"
                            ] 
        }, 

    "ODI": {"training_files": [   
                               "data/odi/Men ODI Player Innings Stats - 2011 to 2020 Team Group 1.csv", 
                               "data/odi/Men ODI Player Innings Stats - 2011 to 2020 Team Group 2.csv", 
                               "data/odi/Men ODI Player Innings Stats - 2011 to 2020 Team Group 3.csv", 
                               "data/odi/Men ODI Player Innings Stats - 2011 to 2020 Team Group 4.csv", 
                               "data/odi/Men ODI Player Innings Stats - 2011 to 2020 Team Group 5.csv", 
                               "data/odi/Men ODI Player Innings Stats - 2021.csv" 
                               ], 
            
            "testing_files": [ 
                            "data/odi/Men ODI Player Innings Stats - 2022.csv",
                            "data/odi/Men ODI Player Innings Stats - 2023.csv",  
                            "data/odi/Men ODI Player Innings Stats - 2024.csv" 
                            ] 
        }, 
}

