import pandas as pd  
from imblearn.over_sampling import SMOTE


from config import PREDICTION_FORMAT

class Util: 
    def __init__(self): 
        pass  

    def generate_confusion_matrix(self, predictions, test_df): 
        if PREDICTION_FORMAT != "BINARY": 
            print("ERROR! Can not generate confusion matrix for non-binary predictions")  
            return dict()
        
        df = test_df
        df["predictions"] = predictions 
    
        statistics = {
            'TP': 0,
            'FP': 0,
            'TN': 0,
            'FN': 0
        }

        for _, row in df.iterrows():
            true_label = row["bucket"]
            predicted_label = row["predictions"]

            if predicted_label == 0 and true_label == 0:
                statistics['TP'] += 1
            elif predicted_label == 0 and true_label == 1:
                statistics['FP'] += 1
            elif predicted_label == 1 and true_label == 1:
                statistics['TN'] += 1
            elif predicted_label == 1 and true_label == 0:
                statistics['FN'] += 1

        # Calculate TPR and TNR
        TPR = statistics['TP'] / (statistics['TP'] + statistics['FN']) if (statistics['TP'] + statistics['FN']) > 0 else 0
        TNR = statistics['TN'] / (statistics['TN'] + statistics['FP']) if (statistics['TN'] + statistics['FP']) > 0 else 0

        # Calculate Accuracy
        accuracy = (statistics['TP'] + statistics['TN']) / (statistics['TP'] + statistics['TN'] + statistics['FP'] + statistics['FN']) if (statistics['TP'] + statistics['TN'] + statistics['FP'] + statistics['FN']) > 0 else 0

        statistics['TPR'] = round(TPR*100,2)
        statistics['TNR'] = round(TNR*100,2)
        statistics['Accuracy'] = round(accuracy*100,2)

        return statistics  
    

    def print_confusion_matrix(self, confusion_matrix:dict): 
        statistics_df = pd.DataFrame([confusion_matrix]) 
        print(statistics_df) 
        print('\n') 
    
    def resample_data_with_smote(self, df, training_features):
        feature_data = df[training_features]
        label_data = df['bucket']

        # Apply SMOTE
        smote = SMOTE(k_neighbors = 5, random_state=42)
        feature_resampled, label_resampled = smote.fit_resample(feature_data, label_data)

        # Combine the resampled features and target into a DataFrame
        df_resampled = pd.DataFrame(feature_resampled, columns=feature_resampled.columns)
        df_resampled['bucket'] = label_resampled 

        return df_resampled


