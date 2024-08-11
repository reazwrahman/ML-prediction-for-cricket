import itertools
import argparse
import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
script_dir = os.path.dirname(os.path.abspath(__file__))

from classifiers.knn_classifer import KNNClassifier
from classifiers.logistic_regression import LogisticRegressionClassifier
from classifiers.random_forest import MyRandomForestClassifier
from classifiers.gbm_classifier import GBMClassifier
from classifiers.svm_classifier import SVMClassifier
from config import PREDICTION_FORMAT, GAME_FORMAT, PLAYER_ROLE, FEATURES


TOP_FEATURE_COUNT = 3  ## SET TO None, if all features are to be used


def analyze_summary(df):
    print("Overall Analysis Summary")
    print("-" * 80)

    # Rank classifiers based on Accuracy
    df["Accuracy Rank"] = df["Accuracy"].rank(ascending=False, method="min")
    best_accuracy_classifier = df.loc[df["Accuracy Rank"] == 1]["Classifier"].values[0]
    best_accuracy = df["Accuracy"].max()
    print(
        f"Best classifier based on Accuracy: {best_accuracy_classifier} with Accuracy = {best_accuracy}%"
    )

    # Rank classifiers based on TPR
    df["TPR Rank"] = df["TPR"].rank(ascending=False, method="min")
    best_tpr_classifier = df.loc[df["TPR Rank"] == 1]["Classifier"].values[0]
    best_tpr = df["TPR"].max()
    print(f"Classifier with highest TPR: {best_tpr_classifier} with TPR = {best_tpr}%")

    # Rank classifiers based on TNR
    df["TNR Rank"] = df["TNR"].rank(ascending=False, method="min")
    best_tnr_classifier = df.loc[df["TNR Rank"] == 1]["Classifier"].values[0]
    best_tnr = df["TNR"].max()
    print(f"Classifier with highest TNR: {best_tnr_classifier} with TNR = {best_tnr}%")

    # Display ranking summary
    print("\nRanking Summary:")
    print(df[["Classifier", "Accuracy Rank", "TPR Rank", "TNR Rank"]])


if __name__ == "__main__":
    top_feature_count: int = TOP_FEATURE_COUNT
    if top_feature_count is None:
        top_feature_count = len(FEATURES)

    print(f"Top {top_feature_count} features used")
    registrar = dict()
    registrar["KNN"] = KNNClassifier
    registrar["LOGISTIC REGRESSION"] = LogisticRegressionClassifier
    registrar["RANDOM FOREST"] = MyRandomForestClassifier
    registrar["GBM"] = GBMClassifier

    dfs = []
    for each in registrar:
        classifier = registrar[each]()
        classifier.build_model(classifier.x_train[classifier.all_features])
        top_features = list(classifier.get_feature_importance()["Feature"])[
            0:top_feature_count
        ]
        classifier.update_features(top_features)
        predictions = classifier.make_predictions()
        accuracy = classifier.compute_accuracy(predictions)
        conf_matrix = classifier.generate_confusion_matrix(predictions)
        conf_matrix["Classifier"] = classifier.name
        conf_df = pd.DataFrame([conf_matrix])
        dfs.append(conf_df)

    print(
        f"GAME FORMAT: {GAME_FORMAT}, PREDICTION FORMAT: {PREDICTION_FORMAT}, PLAYER ROLE: {PLAYER_ROLE}"
    )
    print("\n")
    summary_df = pd.concat(dfs, axis=0)
    print(summary_df)
    avg_accuracy = round(summary_df["Accuracy"].mean(), 2)
    print(f"Average accuracy: {avg_accuracy}")
    analyze_summary(summary_df)
