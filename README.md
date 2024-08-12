# Project Scope 

This repository analyzes a number of machine learning models designed to predict whether a cricket player (batter and bowler) will exceed a specific performance threshold in upcoming matches. The project leverages open source historical data on players' performances in each individual game and extracts the most relevant features including runs scored, strike rates, recent performance in the last 'N' games and match conditions, to train various predictive models such as logistic regression, Random Forest, GBM, KNN and SVM. 

The analysis is focused around a) finding the right feature combinations for each classifier and b) coming up with a ranking system to compare the performance of the classifiers based on Accuracy, TPR and TNR. 

Youtube Presentation Link: https://www.youtube.com/watch?v=IfC2IZdt6nU

# How to run the program 

`Important`: run all programs from the root directory. If you attempt to run the files from their individual directory, you will run into importing errors. 

## Installing packages  
setup a virtual environment with venv, conda env or pipenv and install the 
packages listed in requirements.txt, if your machine is already prepared to
run typical data science projects you most likely have all the required packages 
and are safe to move to the next step. 

Once environment is setup, run: ```pip install -r requirements.txt```

## Configs 

You can change the configs from the configs.py file or keep the default parameters as is to observe the general behavior of the program. 

Each parameter is commented with explanation and supported values. All the data files (training and testing) are defined in the configs.py as well. 

## Cleaned up data with relevant features 

Raw data from the csv are cleaned up and the relevant features are extracted in the data/batting_data.py and data/bowling_data.py.
To run these: ```python3 data/batting_data.py``` 

```python3 data/bowling_data.py``` 

## Running the classifiers 
The classifiers are under the classifiers directory. All classifiers inherit the common functionalities from the BaseClassifier. When these classifiers are run, it will show the accuracy and the confusion matrix for the parameters defined in the configs.py. 

To run the logistic regression for example: 
```python3 classifiers/logistic_regression.py```

## Running analysis

All the analysis programs are under the analysis directory. Example: 
```python3 analysis/performance_analyzer.py```

1) features_kde_plotter: shows the separation of the two classes by each feature in a kde plot. This helps in visualizing how each feature separates the data on its own. 

2) optimal_feature_finder: performs a brute force search over the entire feature set space and finds the feature combination that produces the best TPR, best TNR and best accuracy for a particular classifier. 

3) performance_analyzer: performs a ranking among all the registered classifiers based on TPR, TNR and accuracy value. 

## Viewing Test Artifacts 

Results from the analysis program above, general observations and analysis performed to get the optimal parameters for some of the classifiers are recorded under the test_artifacts directory.  

There is also a Comprehensive Analysis.pdf at the top level discussing all the findings. 

