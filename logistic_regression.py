import copy
import pandas as pd 
from sklearn.linear_model import LogisticRegression  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from batting_data import BattingDataUtil 
from config import GAME_FORMAT

class LogisticRegressionClassifier: 
    def __init__(self): 
        self.batting_util = BattingDataUtil()
        self.x_train = self.batting_util.get_training_data() 
        self.x_test = self.batting_util.get_testing_data()
        self.all_features = self.batting_util.selected_features  

        self.scaler = StandardScaler() 
        self.model = None
        
    def scale_training_data(self):
        self.scaler.fit_transform(self.x_train[self.all_features]) 
    
    def build_model(self, training_data): 
        model = LogisticRegression(random_state=42, max_iter=10000)
        model.fit(training_data, self.x_train["bucket"])   
        return model 
    
    def make_predictions(self): 
        model = self.build_model(self.x_train[self.all_features])
        predictions = model.predict(self.x_test[self.all_features]) 
        return predictions 

    def compute_accuracy(self, predictions): 
        accuracy = accuracy_score(self.x_test["bucket"], predictions)  
        return accuracy*100
    
    def experiment_dropping_feature(self): 
        for i in range (len(self.all_features)): 
            selected_features = copy.deepcopy(self.all_features)   
            print (f'logistic regression::dropping feature {self.all_features[i]}:')
            del(selected_features[i])  

            model = self.build_model(self.x_train[selected_features])
            predictions = model.predict(self.x_test[selected_features])
            accuracy = classifier.compute_accuracy(predictions) 
            print(accuracy) 
            print('\n')
        
    def make_single_prediction(self, features): 
        pass #TODO


if __name__ == "__main__": 
    print(f'GAME FORMAT: {GAME_FORMAT}')
    classifier = LogisticRegressionClassifier()
    accuracy = classifier.compute_accuracy(classifier.make_predictions())  
    print(f'logistic regression all features used')
    print(accuracy)  
    print('\n')
    #classifier.experiment_dropping_feature()