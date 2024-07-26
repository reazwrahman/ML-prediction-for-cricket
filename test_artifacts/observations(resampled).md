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

KNN
GAME FORMAT: ODI, PREDICTION FORMAT: BINARY, PLAYER ROLE: BATTER
Best TPR combination: ('country',) with TPR = 100.0, with TNR = 0.0, with Accuracy = 70.84
Best TNR combination: ('opposition', 'avg_runs') with TNR = 83.51, with TPR = 51.01, with Accuracy = 60.49
Best Accuracy combination: ('country',) with Accuracy = 70.84, with TPR = 100.0, with TNR = 0.0


GAME FORMAT: ODI, PREDICTION FORMAT: BINARY, PLAYER ROLE: BATTER
KNN classifier all features used
63.49498669940659


     TP   FP   TN    FN    TPR    TNR  Accuracy
0  2120  442  983  1342  61.24  68.98     63.49
=================================================
KNN
GAME FORMAT: ODI, PREDICTION FORMAT: BINARY, PLAYER ROLE: BOWLER
Best TPR combination: ('country',) with TPR = 97.32, with TNR = 2.58, with Accuracy = 65.02
Best TNR combination: ('opposition', 'career_wickets_per_game', 'career_strike_rate') with TNR = 88.41, with TPR = 34.69, with Accuracy = 53.01
Best Accuracy combination: ('career_wickets_per_game',) with Accuracy = 68.57, with TPR = 88.65, with TNR = 29.77


GAME FORMAT: ODI, PREDICTION FORMAT: BINARY, PLAYER ROLE: BOWLER
KNN classifier all features used
56.869300911854104


     TP   FP   TN    FN    TPR    TNR  Accuracy
0  1122  373  749  1046  51.75  66.76     56.87


=====================================================================================
LOGISTIC REGRESSION
GAME FORMAT: ODI, PREDICTION FORMAT: BINARY, PLAYER ROLE: BATTER
Best TPR combination: ('recent_avg', 'recent_avg_sr') with TPR = 70.68, with TNR = 54.81, with Accuracy = 66.05
Best TNR combination: ('ground',) with TNR = 100.0, with TPR = 0.0, with Accuracy = 29.16
Best Accuracy combination: ('opposition', 'avg_runs', 'avg_sr') with Accuracy = 66.36, with TPR = 64.53, with TNR = 70.81

GAME FORMAT: ODI, PREDICTION FORMAT: BINARY, PLAYER ROLE: BATTER
logistic regression all features used
66.15510538162472


     TP   FP    TN    FN    TPR    TNR  Accuracy
0  2208  400  1025  1254  63.78  71.93     66.16
==================================================

LOGISTIC REGRESSION
GAME FORMAT: ODI, PREDICTION FORMAT: BINARY, PLAYER ROLE: BOWLER
Best TPR combination: ('career_wickets_per_game', 'recent_wickets_per_game') with TPR = 56.0, with TNR = 76.11, with Accuracy = 62.86
Best TNR combination: ('ground',) with TNR = 100.0, with TPR = 0.0, with Accuracy = 34.1
Best Accuracy combination: ('opposition', 'career_wickets_per_game', 'recent_wickets_per_game') with Accuracy = 62.89, with TPR = 55.95, with TNR = 76.29

GAME FORMAT: ODI, PREDICTION FORMAT: BINARY, PLAYER ROLE: BOWLER
logistic regression all features used
61.7629179331307


     TP   FP   TN    FN    TPR    TNR  Accuracy
0  1148  238  884  1020  52.95  78.79     61.76

================================================
RANDOM FOREST CLASSIFIER
GAME FORMAT: ODI, PREDICTION FORMAT: BINARY, PLAYER ROLE: BATTER
Best TPR combination: ('recent_avg',) with TPR = 65.43, with TNR = 56.01, with Accuracy = 62.81
Best TNR combination: ('country', 'avg_runs', 'recent_avg', 'avg_sr') with TNR = 98.27, with TPR = 24.37, with Accuracy = 44.91
Best Accuracy combination: ('recent_avg',) with Accuracy = 62.81, with TPR = 65.43, with TNR = 56.01

GAME FORMAT: ODI, PREDICTION FORMAT: BINARY, PLAYER ROLE: BATTER
RANDOM FOREST CLASSIFIER all features used
51.41457562731181


     TP   FP    TN    FN    TPR    TNR  Accuracy
0  2587  224  2556  4636  35.82  91.94     51.41