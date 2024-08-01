#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 10:37:55 2024

@author: reazrahman
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# source data: https://data.world/cclayford/cricinfo-statsguru-data
filenames = [
    "Men T20I Player Innings Stats - 2021.csv",
    "Men T20I Player Innings Stats - 2022.csv",
    "Men T20I Player Innings Stats - 2023.csv",
]

filenames = ["Men T20I Player Innings Stats - 2024.csv"]
dfs = []


for i in range(len(filenames)):
    df = pd.read_csv(filenames[i])
    batting_columns = [
        "Innings Player",
        "Innings Runs Scored Num",
        "Innings Not Out Flag",
        "Innings Balls Faced",
        "Innings Batting Strike Rate",
        "Opposition",
        "Ground",
        "Innings Date",
        "Country",
        "Innings Runs Scored Buckets",
    ]

    df = df[batting_columns]
    new_batting_column = [
        "player",
        "runs",
        "not_out_flag",
        "balls_faced",
        "strike_rate",
        "opposition",
        "ground",
        "date",
        "country",
        "bucket",
    ]

    df.columns = new_batting_column
    dfs.append(df)


df = pd.concat(dfs, ignore_index=True)

## cleaning process
df.replace("-", np.nan, inplace=True)
df = df.dropna(subset=["runs"])
df["opposition"] = df["opposition"].str.replace("^v ", "", regex=True)
df["date"] = pd.to_datetime(df["date"])

df["runs"] = df["runs"].astype(int)
df["not_out_flag"] = df["not_out_flag"].astype(int)
df["strike_rate"] = df["strike_rate"].astype(float)


def refine_buckets(runs):
    if runs >= 0 and runs <= 25:
        return "0-25"
    elif runs > 25 and runs <= 50:
        return "25-50"
    elif runs > 50 and runs <= 75:
        return "50-75"
    elif runs > 75 and runs <= 100:
        return "75-100"
    else:
        return "100+"


df["bucket"] = df["runs"].apply(refine_buckets)


## find average
grouped = (
    df.groupby("player")
    .agg(
        total_runs=pd.NamedAgg(column="runs", aggfunc="sum"),
        innings_count=pd.NamedAgg(
            column="not_out_flag", aggfunc=lambda x: (x == 0).sum()
        ),
    )
    .reset_index()
)

grouped["avg_runs"] = round(grouped["total_runs"] / grouped["innings_count"], 2)
# Replace inf values with the total_runs values
grouped["avg_runs"] = grouped.apply(
    lambda row: row["total_runs"] if np.isinf(row["avg_runs"]) else row["avg_runs"],
    axis=1,
)
df = pd.merge(df, grouped[["player", "avg_runs"]], on="player")


## current form (average) for n games

n = 5

# Sort DataFrame by 'Player' and 'Innings Date'
df = df.sort_values(by=["player", "date"], ascending=[True, False])

# Group by 'Player' and take the latest 'n' matches
df_latest_n = df.groupby("player").head(n)

# Group by 'Player' and calculate total runs and valid innings count
grouped = (
    df_latest_n.groupby("player")
    .agg(
        total_runs=pd.NamedAgg(column="runs", aggfunc="sum"),
        innings_count=pd.NamedAgg(
            column="not_out_flag", aggfunc=lambda x: (x == 0).sum()
        ),
    )
    .reset_index()
)

# Calculate average runs
grouped["recent_avg"] = grouped["total_runs"] / grouped["innings_count"]

# Replace inf values with the total_runs values
grouped["recent_avg"] = grouped.apply(
    lambda row: row["total_runs"] if np.isinf(row["recent_avg"]) else row["recent_avg"],
    axis=1,
)

# Merge the average runs back into the original DataFrame
df = pd.merge(df, grouped[["player", "recent_avg"]], on="player", how="left")


## replace nan strike rate to a reasonable value
df["strike_rate"] = df.apply(
    lambda row: (
        row["runs"] * 100 if pd.isna(row["strike_rate"]) else row["strike_rate"]
    ),
    axis=1,
)


## average strike rate
df["strike_rate"] = pd.to_numeric(df["strike_rate"], errors="coerce")
average_sr = df.groupby("player")["strike_rate"].mean().reset_index()
average_sr.columns = ["player", "avg_sr"]
df = pd.merge(df, average_sr, on="player")
df["avg_sr"] = round(df["avg_sr"], 2)

## recent strike rate
recent_avg_sr = df_latest_n.groupby("player")["strike_rate"].mean().reset_index()
recent_avg_sr.columns = ["player", "recent_avg_sr"]
df = pd.merge(df, recent_avg_sr, on="player")
df["recent_avg_sr"] = round(df["recent_avg_sr"], 2)


## add year
df["year"] = df["date"].apply(lambda row: row.year)


# one-hot encoding
le = LabelEncoder()
df["country_encoded"] = le.fit_transform(df["country"])
df["bucket_encoded"] = le.fit_transform(df["bucket"])

print(df)
print(df.head(50))
print(df.tail(50))
print(df["country"].unique())
