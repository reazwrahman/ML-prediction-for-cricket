# Some observations 

1) both KNN and logistic regression have very comparable accuracy, KNN winning 
by a very thin margin 

2) However, logistic regression have a much better rate at predicting 
True Negatives. For some experiments, KNN was simply unable to detect 
any True Negative at all (scoring above the threshold).  

My understanding is this is due to an overwhelming number of postive neighbor 
around (below the threshold). 

For this use case, I would like to see a higher TNR rate. Therefore, between 
these two classifiers I would pick logistic regression despite KNN having a 
slightly higher accuracy. 

3) My observation holds true for both ODI and T20 

4) I tried playing around with the features: using two sets of features, one
with just performance stats (avg/strike rate), other with categorical 
information (ground, opposition, country). I did not see any pattern that 
will challenge my conclusion above. 


5) I wrote a little program to see which set of features produced the  
best TPR, TNR and Accuracy. Recording the results below: 

KNN CLASSIFIER:
GAME FORMAT: T20, PREDICTION FORMAT: BINARY, PLAYER ROLE: BATTER
Best TPR combination: ('ground',) with TPR = 100.0, with TNR = 0.0, with Accuracy = 79.4
Best TNR combination: ('avg_runs', 'recent_avg', 'avg_sr', 'recent_avg_sr') with TNR = 5.54, with TPR = 98.68, with Accuracy = 79.5
Best Accuracy combination: ('ground', 'country', 'avg_runs', 'avg_sr') with Accuracy = 79.57, with TPR = 99.65, with TNR = 2.15 

GAME FORMAT: T20, PREDICTION FORMAT: BINARY, PLAYER ROLE: BATTER
KNN classifier all features used
79.3103448275862


     TP   FP  TN  FN    TPR   TNR  Accuracy
0  3399  879   5   9  99.74  0.57     79.31

KNN CLASSIFIER:
GAME FORMAT: T20, PREDICTION FORMAT: BINARY, PLAYER ROLE: BOWLER
Best TPR combination: ('ground',) with TPR = 100.0, with TNR = 0.0, with Accuracy = 71.41
Best TNR combination: ('opposition', 'career_wickets_per_game', 'career_strike_rate') with TNR = 18.37, with TPR = 91.63, with Accuracy = 70.68
Best Accuracy combination: ('career_wickets_per_game',) with Accuracy = 72.7, with TPR = 96.34, with TNR = 13.64

GAME FORMAT: T20, PREDICTION FORMAT: BINARY, PLAYER ROLE: BOWLER
KNN classifier all features used
71.34318968229003


     TP   FP  TN  FN    TPR   TNR  Accuracy
0  2262  903   6   8  99.65  0.66     71.34

============================================================================= 
LOGISTIC REGRESSION CLASSIFIER:
GAME FORMAT: T20, PREDICTION FORMAT: BINARY, PLAYER ROLE: BATTER
Best TPR combination: ('opposition',) with TPR = 100.0, with TNR = 0.0, with Accuracy = 79.4
Best TNR combination: ('opposition', 'avg_runs', 'avg_sr') with TNR = 14.14, with TPR = 95.63, with Accuracy = 78.84
Best Accuracy combination: ('opposition',) with Accuracy = 79.4, with TPR = 100.0, with TNR = 0.0 

GAME FORMAT: T20, PREDICTION FORMAT: BINARY, PLAYER ROLE: BATTER
logistic regression all features used
78.68126747437091


     TP   FP   TN   FN   TPR    TNR  Accuracy
0  3258  765  119  150  95.6  13.46     78.68

LOGISTIC REGRESSION CLASSIFIER:
GAME FORMAT: T20, PREDICTION FORMAT: BINARY, PLAYER ROLE: BOWLER
Best TPR combination: ('opposition',) with TPR = 100.0, with TNR = 0.0, with Accuracy = 71.41
Best TNR combination: ('opposition', 'ground', 'country', 'career_wickets_per_game', 'recent_wickets_per_game') with TNR = 16.5, with TPR = 95.68, with Accuracy = 73.04
Best Accuracy combination: ('ground', 'career_wickets_per_game', 'recent_wickets_per_game', 'recent_strike_rate') with Accuracy = 73.17, with TPR = 96.92, with TNR = 13.86 

GAME FORMAT: T20, PREDICTION FORMAT: BINARY, PLAYER ROLE: BOWLER
logistic regression all features used
72.8845548914753

     TP   FP   TN  FN    TPR    TNR  Accuracy
0  2181  773  136  89  96.08  14.96     72.88

