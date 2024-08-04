# Project Scope 

TODO

# How to run the program 
`Important`: run all programs from the root directory. If you attempt to run the files from their individual directory, you will run into importing errors. 

### Installing packages  
setup a virtual environment with venv, conda env or pipenv and install the 
packages listed in requirements.txt, if your machine is already prepared to
run typical data science projects you most likely have all the required packages 
and are safe to move to the next step. 

### Configs 

You can change the configs from the configs.py file or keep the default parameters as is to observe the general behavior of the program. 

Each parameter is commented with explanation and supported values. All the data files (training and testing) are defined in the configs.py as well. 

### Cleaned data with relevant features 

Raw data from the csv are cleaned up and the relevant features are extracted in the data/batting_data.py and data/bowling_data.py.
To run these: ```python3 data/batting_data.py``` 

```python3 data/bowling_data.py``` 

### Running the classifiers 
The classifiers are under the classifiers directory. All classifiers inherit the common functionalities from the BaseClassifier. When these classifiers are run, it will show the accuracy and the confusion matrix for the parameters defined in the configs.py. 

To run the logistic regression for example: 
```python3 classifiers/logistic_regression.py```

### Running analysis

All the analysis programs are under the analysis directory. 

1) features_kde_plotter: shows the separation of the two classes by each feature in a kde plot. This helps in visualizing how each feature separates the data on its own. 

2) optimal_feature_finder: performs a brute search over the entire feature set space and finds the feature combination that produces the best TPR, best TNR and best accuracy for a particular classifier. 

3) performance_analyzer: performs a ranking among all the registered classifiers based on TPR, TNR and accuracy value. 


