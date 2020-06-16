# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:00:01 2020

@author: Værksted Matilde
"""

from sklearn.ensemble import RandomForestClassifier
#from POST import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
#from Equal_opportunity import equal_opportunity 
#from conf_and_rates import plot_conf
import numpy as np
import pickle
from PIL import Image  
import PIL  
from sklearn.model_selection import RandomizedSearchCV
from Process_data import A, ytrue, yhat
from Process_data import y_train, y_test, X_train, X_test, train_index, test_index
import pandas as pd


np.random.seed(217)
#%% Functions

def load_classifier_1(name, X_train, y_train):

    if name == "NN":

        model = load_model("./NN_model.h5")

    elif name == "RF":
        model = RandomForestClassifier(n_estimators=65,
                           criterion = 'gini',
                           bootstrap = True,
                           max_features =7,
                           ccp_alpha =0.0011)
    
        model.fit(X_train, y_train)    
    
        
    else:
        print("Wrong model name")

    return model


def train_test_RF(X_train, y_train, X_test, y_test): 
    
    model = load_classifier_1('RF', X_train, y_train)
    # Probabilities for score = 1, test
    yhat_rf = model.predict_proba(X_test)[:, 1]
    yhat_rf = pd.DataFrame(yhat_rf)
    
    #Training and test accurracy
    train_acc = model.score(X_train, y_train)
    test_acc =  model.score(X_test, y_test)
    
    return train_acc, test_acc, yhat_rf, model
    
    
#%%

#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

def CV_RF(): 
    
    #rf_random: return RandomizedSearchCV of randomforest
        
    #Define hyper parameter space
    n_estimators=[int(x) for x in np.arange(64,70,1)]
    criterion = ['entropy', 'gini']
    max_features = [int(x) for x in np.arange(4,10,1)]
    ccp_alpha = [x for x in np.arange(0.001, 0.1, 0.0001)]
    
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features, 
                   'criterion': criterion, 
                   'ccp_alpha': ccp_alpha}
    
    #Define model
    rf = RandomForestClassifier()
    
    #Run CV
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 2000, cv = 10, verbose=2, random_state=42, n_jobs = -1, return_train_score = True)
    rf_random.fit(X_train, y_train)

        
    return rf_random

#%%
#Uncomment to run CV 
"""
rf_random = CV_RF()

#Access best parameters, best test-score (mean across cv folds) and best model
best_param = rf_random.best_params_
best_score = rf_random.best_score_
model = rf_random.best_estimator_

#Access results from CV
cvresults = pd.DataFrame(rf_random.cv_results_)

#mean test and train score of all hyperparameter combinations across folds
train_mean =rf_random.cv_results_['mean_train_score'] 
test_mean = rf_random.cv_results_['mean_test_score'] 

#index of best model
optimal_idx = np.argmax(test_mean)

#Access train and test acc of best model, for all folds
optimal_train = []
optimal_test = []
splits_test = ['split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'split5_test_score', 'split6_test_score', 'split7_test_score', 'split8_test_score', 'split9_test_score']
splits_train = ['split0_train_score', 'split1_train_score', 'split2_train_score', 'split3_train_score', 'split4_train_score', 'split5_train_score', 'split6_train_score' , 'split7_train_score', 'split8_train_score', 'split9_train_score']
for i in splits_train:
    optimal_train.append(rf_random.cv_results_[i][optimal_idx])
for j in splits_test:    
    optimal_test.append(rf_random.cv_results_[j][optimal_idx])
   
#plot test and train score for all folds
plt.plot(optimal_train, '-', label = 'Train accuracy')
plt.plot(optimal_test, '-', label = 'Test accuracy')
plt.plot(best_score, '*', label = 'Mean test accuracy')
plt.plot(np.mean(optimal_train), '*', label = 'Mean train accuracy')
plt.title('Cross validation accuracy of optimal model')
plt.legend(loc = 'lower right', fontsize = 'small')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0,10), np.arange(1,11))
plt.show()

#plot test and train score for all combinaitons of hyperparameters
plt.plot(np.arange(1,len(train_mean)+1,1), train_mean,'.', label = 'Train accuracy')
plt.plot(np.arange(1,len(test_mean)+1,1), test_mean,'.', label = 'Test accuracy')
plt.plot(best_score, '*', label = 'Mean test accuracy')
plt.plot(np.mean(optimal_train), '*', label = 'Mean train accuracy')
plt.title('Mean accuracy (CV) per hyperparameter combination')
plt.xlabel('Hyperparameter combination')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right', fontsize = 'small')
plt.xticks(np.arange(0,len(train_mean)+1), np.arange(1,len(train_mean)+2))
plt.show()

print('Mean test accu: %s' %best_score)
print('Mean train accu: %s' %np.mean(optimal_train))
print(best_param )
"""
