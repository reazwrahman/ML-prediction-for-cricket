import itertools  
import argparse

from knn_classifer import KNNClassifier 
from logistic_regression import LogisticRegressionClassifier
from config import PREDICTION_FORMAT, GAME_FORMAT, PLAYER_ROLE, FEATURES 


parser = argparse.ArgumentParser(description="Enter a classifier type")  
parser.add_argument('value', type=str, help='enter a classifier type, valid values: knn/log/rf/gbm')
args = parser.parse_args()

CLASSIFIER = args.value

registrar = dict() 
registrar['knn'] = KNNClassifier 
registrar['log'] = LogisticRegressionClassifier  

if CLASSIFIER not in registrar: 
    raise Exception('Invalid Classifer Name')


def generate_combinations(features):
    all_combinations = []
    for r in range(1, len(features) + 1):
        combinations = list(itertools.combinations(features, r))
        all_combinations.extend(combinations)
    return all_combinations

all_feature_combinations = generate_combinations(FEATURES) 
#all_feature_combinations = all_feature_combinations[0:2]

# Initialize dictionary to store stats
stats_dict = {}

# Assuming classifier is already defined and has the methods used below 
counter = 1 
for combination in all_feature_combinations:
    features = list(combination) 
    classifier = registrar[CLASSIFIER]()
    classifier.update_features(features)
    predictions = classifier.make_predictions()
    stats = classifier.generate_confusion_matrix(predictions)
    
    tpr = stats['TPR']
    tnr = stats['TNR'] 
    accuracy = stats['Accuracy']
    
    stats_dict[tuple(features)] = {
        'TPR': tpr,
        'TNR': tnr,
        'Accuracy': accuracy
    } 
    print(counter) 
    counter +=1

# Find the best combinations  
print (f'{registrar[CLASSIFIER]().name}')
print(f'GAME FORMAT: {GAME_FORMAT}, PREDICTION FORMAT: {PREDICTION_FORMAT}, PLAYER ROLE: {PLAYER_ROLE}')
best_tpr_combination = max(stats_dict, key=lambda x: stats_dict[x]['TPR'])
best_tnr_combination = max(stats_dict, key=lambda x: stats_dict[x]['TNR'])
best_accuracy_combination = max(stats_dict, key=lambda x: stats_dict[x]['Accuracy'])

print(f"Best TPR combination: {best_tpr_combination} with TPR = {stats_dict[best_tpr_combination]['TPR']}, with TNR = {stats_dict[best_tpr_combination]['TNR']}, with Accuracy = {stats_dict[best_tpr_combination]['Accuracy']}")

print(f"Best TNR combination: {best_tnr_combination} with TNR = {stats_dict[best_tnr_combination]['TNR']}, with TPR = {stats_dict[best_tnr_combination]['TPR']}, with Accuracy = {stats_dict[best_tnr_combination]['Accuracy']}") 

print(f"Best Accuracy combination: {best_accuracy_combination} with Accuracy = {stats_dict[best_accuracy_combination]['Accuracy']}, with TPR = {stats_dict[best_accuracy_combination]['TPR']}, with TNR = {stats_dict[best_accuracy_combination]['TNR']}")


