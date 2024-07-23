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

