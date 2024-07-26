1) cleaning up the original data to have the features that I think 
will have the most impact on prediction 

2) comparing results between KNN and logistic regression 
- finding optimal neighbor 
- creating the optimal feature finder algorithm and its results 

3) saw a major imbalance in the training data as i was investigating 
the low TNR issue 
bowling data:    ## label=1, percentage in training data 27%(TN), label=0, 73% (TP) 
batting data:    ## label=1, percentage in training data 20%(TN), label=0, 80% (TP) 

- started looking into synthetic data creation to overcome this balance 
- found SMOTE and applied it to balance data equally 

4) Reobserved optimal feature finder result 

