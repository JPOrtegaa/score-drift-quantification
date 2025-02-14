import mlquantify as mq

mq.set_arguments(y_pred=None, # Predictions for the TEST set ()
                 posteriors_test=None, # Scores of the TEST set ()
                 y_labels=None, # True labels of the TRAIN set ()
                 y_pred_train=None, # Predictions extracted from TRAIN set ()
                 posteriors_train=None # Scores extracted from TRAIN set ()
            )

"""
    Mostly of probabilistic methods must pass:          # DyS, DySsyn, SORD, HDy, EMQ, FM, SMM, All threshold methods
        - y_pred 
        - posteriors_test
        - y_labels
        = posteriors_train
    
    Methods like CC, e PWK só precisam de:
        - y_pred
    
    GAC and GPAC methods must have:
        - y_pred_train
        - y_pred
        
        
    You don't have to put all the arguments, just the ones that are necessary for the method you are going to use.
    The arguments that are not necessary for the method will be ignored.
    
    If you pass any arguments, you must put learner on the quantification methods to None, for example:
    
    DyS = mq.DyS(learner=None) or DyS = mq.DyS()
    
    and call fit and predict the normal way.
    
    dys.fit(X_train, y_train)
    predict = dys.predict(X_test)
"""