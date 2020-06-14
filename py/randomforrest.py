# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:00:01 2020

@author: Værksted Matilde
"""

from sklearn.ensemble import RandomForestClassifier
from POST import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
#from Equal_opportunity import equal_opportunity 
#from conf_and_rates import plot_conf
import numpy as np
import pickle
#from Permutation_test import load_classifier

np.random.seed(217)
#%% Functions

def load_classifier_1(name, X_train, y_train):

    if name == "NN":

        model = load_model("./NN_model.h5")

    elif name == "RF":
        model = RandomForestClassifier(n_estimators=64,
                               criterion = 'entropy',
                               min_samples_split=2,
                               bootstrap = True,
                               max_features = None)
        
        model.fit(X_train, y_train)
    
        
    else:
        print("Wrong model name")

    return model

def train_test_RF(X_train, y_train, X_test, y_test, train = True): 
    
# Fit on training data
    if train == True: 
        model = RandomForestClassifier(n_estimators=64,
                                       #max_depth = 15,
                               criterion = 'entropy',
                               #min_samples_split=50,
                               #min_samples_leaf = 30,
                               bootstrap = True,
                               max_features = 5, 
                               oob_score = True, 
                               ccp_alpha =0.0029)
#0.0078
        
        model.fit(X_train, y_train)
    
    else: 
        model = load_classifier_1("RF", X_train, y_train)
    
   
    # Probabilities for score = 1, test
    yhat_rf = model.predict_proba(X_test)[:, 1]
    yhat_rf = pd.DataFrame(yhat_rf)
    
    #Training and test accurracy
    train_acc = model.score(X_train, y_train)
    test_acc =  model.score(X_test, y_test)
    
    return train_acc, test_acc, yhat_rf, model
    
#%%
#Uncomment to train model 
    

# Define random forrest model





#Import variables from other scripts
#from Process_data import A, ytrue, yhat
#from Process_data import y_train, y_test, X_train, X_test, train_index, test_index

#train and test model
#train_acc, test_acc, yhat_rf, model = train_test_RF(X_train, y_train, X_test, y_test)
          


#save model
#filename = 'RF.sav'
#pickle.dump(model, open(filename, 'wb'))




