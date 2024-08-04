import itertools
import argparse
import os
import sys
import multiprocessing
from functools import partial

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
script_dir = os.path.dirname(os.path.abspath(__file__))

from classifiers.knn_classifer import KNNClassifier
from classifiers.logistic_regression import LogisticRegressionClassifier
from classifiers.random_forest import MyRandomForestClassifier 
from classifiers.gbm_classifier import GBMClassifier
from config import PREDICTION_FORMAT, GAME_FORMAT, PLAYER_ROLE, FEATURES


def generate_combinations(features):
    all_combinations = []
    for r in range(1, len(features) + 1):
        combinations = list(itertools.combinations(features, r))
        all_combinations.extend(combinations)
    return all_combinations


def process_classifier(classifier_key, all_feature_combinations, registrar):
    stats_dict = {}
    counter = 0

    for combination in all_feature_combinations:
        classifier = registrar[classifier_key]()
        features = list(combination)
        classifier.update_features(features)
        predictions = classifier.make_predictions()
        stats = classifier.generate_confusion_matrix(predictions)

        tpr = stats["TPR"]
        tnr = stats["TNR"]
        accuracy = stats["Accuracy"]

        stats_dict[tuple(features)] = {"TPR": tpr, "TNR": tnr, "Accuracy": accuracy}
        counter += 1

    # Find the best combinations
    best_tpr_combination = max(stats_dict, key=lambda x: stats_dict[x]["TPR"])
    best_tnr_combination = max(stats_dict, key=lambda x: stats_dict[x]["TNR"])
    best_accuracy_combination = max(stats_dict, key=lambda x: stats_dict[x]["Accuracy"])
    print("-" * 80)
    print(f"classifier: {classifier_key}")
    print(
        f"Best TPR combination: {best_tpr_combination} with TPR = {stats_dict[best_tpr_combination]['TPR']}, with TNR = {stats_dict[best_tpr_combination]['TNR']}, with Accuracy = {stats_dict[best_tpr_combination]['Accuracy']}"
    )

    print(
        f"Best TNR combination: {best_tnr_combination} with TNR = {stats_dict[best_tnr_combination]['TNR']}, with TPR = {stats_dict[best_tnr_combination]['TPR']}, with Accuracy = {stats_dict[best_tnr_combination]['Accuracy']}"
    )

    print(
        f"Best Accuracy combination: {best_accuracy_combination} with Accuracy = {stats_dict[best_accuracy_combination]['Accuracy']}, with TPR = {stats_dict[best_accuracy_combination]['TPR']}, with TNR = {stats_dict[best_accuracy_combination]['TNR']}"
    )
    print("\n")


if __name__ == "__main__":
    print(
        f"GAME FORMAT: {GAME_FORMAT}, PREDICTION FORMAT: {PREDICTION_FORMAT}, PLAYER ROLE: {PLAYER_ROLE}"
    )
    all_feature_combinations = generate_combinations(FEATURES)
    ##all_feature_combinations = all_feature_combinations[0:2]  # for quick testing

    registrar = dict()
    registrar["KNN"] = KNNClassifier
    registrar["LOGISTIC REGRESSION"] = LogisticRegressionClassifier
    registrar["RANDOM FOREST"] = MyRandomForestClassifier 
    registrar["GBM"] = GBMClassifier

    # Multiprocessing pool
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(
            process_classifier,
            [
                (classifier, all_feature_combinations, registrar)
                for classifier in registrar.keys()
            ],
        )
