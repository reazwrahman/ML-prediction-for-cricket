#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 08:40:36 2024

@author: reazrahman
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import copy
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
script_dir = os.path.dirname(os.path.abspath(__file__))

from config import DATA_FILES, GAME_FORMAT, PREDICTION_FORMAT


class BowlingDataGenerator:
    def __init__(self, filenames):
        self.filenames = filenames
        self.df = None
        self.n = 5  ## look at recent n matches for current form
        self.initialize()

    def initialize(self):
        ## procedures
        self.generate_raw_dfs()
        self.do_basic_cleanup()
        self.calculate_wickets_per_game()
        self.calculate_recent_wickets_per_game()
        self.calculate_career_strike_rate()
        self.calculate_recent_strike_rate()
        self.apply_bucket()

    def generate_raw_dfs(self):
        self.raw_dfs = []
        for i in range(len(self.filenames)):
            df = pd.read_csv(self.filenames[i])
            bowling_columns = [
                "Innings Player",
                "Opposition",
                "Ground",
                "Innings Date",
                "Country",
                "Innings Overs Bowled",
                "Innings Bowled Flag",
                "Innings Wickets Taken",
                "Innings Wickets Taken Buckets",
                "Innings Economy Rate",
            ]

            df = df[bowling_columns]
            new_bowling_column = [
                "player",
                "opposition",
                "ground",
                "date",
                "country",
                "overs",
                "bowled_flag",
                "wickets",
                "bucket",
                "economy_rate",
            ]

            df.columns = new_bowling_column
            df = df[df["overs"] != "DNB"]
            df = df[df["overs"] != "TDNB"]
            df = df[df["overs"] != "sub"]
            df = df[df["wickets"] != "Nan"]
            df = df.dropna()
            self.raw_dfs.append(df)
            self.df = pd.concat(self.raw_dfs, ignore_index=True)

    def do_basic_cleanup(self):
        self.df.replace("-", np.nan, inplace=True)
        self.df["opposition"] = self.df["opposition"].str.replace("^v ", "", regex=True)
        self.df["date"] = pd.to_datetime(self.df["date"])

        self.df["overs"] = self.df["overs"].apply(
            self.__convert_overs
        )  ## convert overs to decimal system
        self.df["overs"] = self.df["overs"].astype(float)
        self.df["wickets"] = self.df["wickets"].astype(int)
        self.df["economy_rate"] = self.df["economy_rate"].astype(float)

        self.df["year"] = self.df["date"].apply(lambda row: row.year)
        self.df = self.df.dropna()

    def calculate_wickets_per_game(self):
        grouped = (
            self.df.groupby("player")
            .agg(
                total_wickets=pd.NamedAgg(column="wickets", aggfunc="sum"),
                innings_count=pd.NamedAgg(
                    column="bowled_flag", aggfunc=lambda x: (x == 1).sum()
                ),
            )
            .reset_index()
        )

        grouped["career_wickets_per_game"] = round(
            grouped["total_wickets"] / grouped["innings_count"], 2
        )

        self.df = pd.merge(
            self.df, grouped[["player", "career_wickets_per_game"]], on="player"
        )
        self.df["career_wickets_per_game"] = pd.to_numeric(
            self.df["career_wickets_per_game"], errors="coerce"
        )

    def calculate_recent_wickets_per_game(self):
        # Sort DataFrame by 'Player' and 'Innings Date'
        self.df = self.df.sort_values(by=["player", "date"], ascending=[True, False])

        # Group by 'Player' and take the latest 'n' matches
        df_latest_n = self.df.groupby("player").head(self.n)

        # Group by 'Player' and calculate total wickets and valid innings count
        grouped = (
            df_latest_n.groupby("player")
            .agg(
                total_wickets=pd.NamedAgg(column="wickets", aggfunc="sum"),
                innings_count=pd.NamedAgg(
                    column="bowled_flag", aggfunc=lambda x: (x == 1).sum()
                ),
            )
            .reset_index()
        )

        # Calculate average wickets per game
        grouped["recent_wickets_per_game"] = round(
            grouped["total_wickets"] / grouped["innings_count"], 2
        )

        # Merge the average wickets per game back into the original DataFrame
        self.df = pd.merge(
            self.df,
            grouped[["player", "recent_wickets_per_game"]],
            on="player",
            how="left",
        )

    def calculate_career_strike_rate(self):
        grouped = (
            self.df.groupby("player")
            .agg(
                total_wickets=pd.NamedAgg(column="wickets", aggfunc="sum"),
                balls_bowled=pd.NamedAgg(column="overs", aggfunc="sum"),
            )
            .reset_index()
        )

        grouped["balls_bowled"] *= 6

        grouped["career_strike_rate"] = round(
            grouped["balls_bowled"] / grouped["total_wickets"], 2
        )

        self.df = pd.merge(
            self.df, grouped[["player", "career_strike_rate"]], on="player"
        )
        self.df["career_strike_rate"] = pd.to_numeric(
            self.df["career_strike_rate"], errors="coerce"
        )

        # Replace inf values
        max_sr = self.df["career_strike_rate"].replace([np.inf, -np.inf], np.nan).max()
        self.df["career_strike_rate"] = self.df["career_strike_rate"].replace(
            np.inf, max_sr
        )

    def calculate_recent_strike_rate(self):
        # Sort DataFrame by 'Player' and 'Innings Date'
        self.df = self.df.sort_values(by=["player", "date"], ascending=[True, False])

        # Group by 'Player' and take the latest 'n' matches
        df_latest_n = self.df.groupby("player").head(self.n)

        grouped = (
            df_latest_n.groupby("player")
            .agg(
                total_wickets=pd.NamedAgg(column="wickets", aggfunc="sum"),
                balls_bowled=pd.NamedAgg(column="overs", aggfunc="sum"),
            )
            .reset_index()
        )

        grouped["balls_bowled"] *= 6

        grouped["recent_strike_rate"] = round(
            grouped["balls_bowled"] / grouped["total_wickets"], 2
        )

        # Merge the average wickets per game back into the original DataFrame
        self.df = pd.merge(
            self.df, grouped[["player", "recent_strike_rate"]], on="player", how="left"
        )

        # Replace inf values
        max_sr = self.df["career_strike_rate"].replace([np.inf, -np.inf], np.nan).max()
        self.df["recent_strike_rate"] = self.df["recent_strike_rate"].replace(
            np.inf, max_sr
        )

    def apply_bucket(self):
        if PREDICTION_FORMAT == "BINARY":
            self.df["bucket"] = self.df["wickets"].apply(self.__use_binary_classifier)
        elif PREDICTION_FORMAT == "CUSTOM":
            self.df["bucket"] = self.df["wickets"].apply(self.__use_custom_classifier)

    def __use_custom_classifier(self, wickets):
        if wickets >= 0 and wickets <= 2:
            return 0
        elif wickets >= 3 and wickets <= 5:
            return 1
        else:
            return 3

    def __use_binary_classifier(self, wickets):
        if wickets >= 0 and wickets < 2:
            return 0  # don't pick
        else:
            return 1  # pick

    def __convert_overs(self, overs):
        over_parts = overs.split(".")
        whole_overs = int(over_parts[0])
        balls = int(over_parts[1])
        return whole_overs + round((balls / 6), 2)


class BowlingDataUtil:
    def __init__(self):
        self.testing_files = DATA_FILES[GAME_FORMAT]["testing_files"]
        self.training_files = DATA_FILES[GAME_FORMAT]["training_files"]
        self.training_data_generator = BowlingDataGenerator(
            filenames=self.training_files
        )
        self.testing_data_generator = BowlingDataGenerator(filenames=self.testing_files)

        self.training_df = self.training_data_generator.df
        self.testing_df = self.training_data_generator.df

        self.selected_features = [
            "opposition",
            "ground",
            "country",
            "career_wickets_per_game",
            "recent_wickets_per_game",
            "career_strike_rate",
            "recent_strike_rate",
        ]

        self.encoding_map = dict()
        self.decoding_map = dict()

        self.initialize()

    def get_all_features(self):
        return copy.deepcopy(self.selected_features)

    def get_training_data(self):
        return self.training_df

    def get_testing_data(self):
        return self.testing_df

    def initialize(self):
        self.training_df = self.__encode_data(self.training_data_generator.df)
        self.testing_df = self.__encode_data(self.testing_data_generator.df)

    def __encode_data(self, df):
        ## encode features
        le = LabelEncoder()
        features = ["opposition", "ground", "country", "bucket"]
        for each in features:
            df[each] = le.fit_transform(df[each])
            self.encoding_map[each] = {
                label: index for index, label in enumerate(le.classes_)
            }
            self.decoding_map[each] = {
                index: label for index, label in enumerate(le.classes_)
            }
        return df

    def get_encode_decode_map(self):
        return {"encoding_map": self.encoding_map, "decoding_map": self.decoding_map}


if __name__ == "__main__":
    bowling_util = BowlingDataUtil()
    print("training data below: ")
    print(bowling_util.get_training_data())
    print("\n")

    print("testing data below: ")
    print(bowling_util.get_testing_data())
    print("\n")

    encoding_decoding_map = bowling_util.get_encode_decode_map()
    # print(encoding_decoding_map["encoding_map"])
    # print(encoding_decoding_map["decoding_map"])
    ## label=1, percentage in training data 27%(TN), label=0, 73% (TP)
    df = bowling_util.training_df
