# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:16:47 2020

@author: Værksted Matilde
"""

from sklearn.ensemble import RandomForestClassifier
from POST import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Equal_opportunity import equal_opportunity 
from conf_and_rates import plot_conf
import numpy as np
from randomforrest import train_test_RF
from tensorflow.keras.models import load_model
import pickle
from Permutation_test import load_classifier

from Equalised_odds import equal_odds, estimate, percentile

#Import variables from other scripts
from Process_data import A, ytrue, yhat
from Process_data import y_train, y_test, X_train, X_test, train_index, test_index

#Prepair A anf ytrue for models
A = A.values[test_index]
A = pd.DataFrame(A)
ytrue = ytrue.values[test_index]
ytrue = pd.DataFrame(ytrue)

#thresholds and sigma
T = np.arange(0,1.001,0.001)
sigma = 0.001


#%% Random forrest
model_rf = load_classifier("RF")
train_acc, test_acc, yhat_rf = train_test_RF(model_rf, X_train, y_train, X_test, y_test, train = False)
print("Training accuracy, RF: %s" %train_acc)
print("Test accuracy, RF:     %s" %test_acc)

#Define equal class variable with two_year_recid[test_index], race[test_index] and predictions on test set
Equal_rf = equal(A[0], yhat_rf[0], ytrue[0], N=400)

#Equalised odds
group = 'African-American'
p0 = [1,1]
FPR_TPR_odds, ACC, conf_odds, tA_odds, tC_odds= equal_odds(T, Equal_rf, group, p0, plot = True)

#Equal opportunity
max_acc, t_odds, FPR_TPR_opp,conf_before, conf_after, acc_before, acc_after, rate_before= equal_opportunity(sigma, T, Equal_rf, plot = True)
 

#Define number of observations in each class
n_A = Equal_rf.Freq[0]
n_C = Equal_rf.Freq[1]

#weighted relative acc
w_acc_odd = (ACC[0]*n_A + ACC[1]*n_C)/(n_A+n_C)
w_acc_opp = (acc_after[0]*n_A + acc_after[1]*n_C)/(n_A+n_C)

#%% NN
model_nn = load_classifier("NN")
y_nn = model_nn.predict(X_test)
loss_nn_test, acc_nn_test = model_nn.evaluate(X_test, y_test, verbose = 0)
loss_nn_train, acc_nn_train = model_nn.evaluate(X_train, y_train, verbose = 0)

print("Test accuracy, NN:     %s" %acc_nn_test)
print("Test loss, NN:     %s" %loss_nn_test)
print("Training accuracy, NN: %s" %acc_nn_train)
print("Training loss, NN: %s" %loss_nn_train)


#Compute FPR and TPR of NN. And define class variable of equal class. 
equal_NN = equal(A[0],y_nn,ytrue[0], N =400)

#Equalised odds
group = 'African-American'
p0 = [1,1]
FPR_TPR_odds_nn, accNN, conf_odds_nn, tAodds_nn, tCodds_nn = equal_odds(T, equal_NN, group, p0, plot = True)

#Equal opportunity
max_acc_nn, t_odds_nn, FPR_TPR_opp_nn,conf_before_nn, conf_opp_nn, acc_before_nn, acc_opp_nn, rate_before_nn = equal_opportunity(sigma, T, equal_NN, plot = True)

#weighted relative acc
w_acc_odd_nn = (accNN[0]*n_A + accNN[1]*n_C)/(n_A+n_C)
w_acc_opp_nn = (acc_opp_nn[0]*n_A + acc_opp_nn[1]*n_C)/(n_A+n_C)
#%% Print results
title = ["RF, Before, A", "RF, Before, C", "RF, odds, A", "RF, odds, C", "RF, opp, A","RF, opp, C",
         "NN, Before, A", "NN, Before, C", "NN, odds, A", "NN, odds, C", "NN, opp, A","NN, opp, C"]
print("RANDOM FORREST")
print("                 ")
print("BEFORE CORRECTING FOR BIAS")
print("                 ")
print("Accuracy:")
print("African-American: %s" %acc_before[0])
print("Caucasian: %s" %acc_before[1])
print("                 ")
print("                 ")
print("FPR and TPR:")
print("African-American")
print("FPR: %s" %rate_before[0][0])
print("TPR: %s" %rate_before[0][1])
print("Caucasian")
print("FPR: %s" %rate_before[1][0])
print("TPR: %s" %rate_before[1][1])
print("                 ")
print("                 ")
print("                 ")
plot_conf(conf_before[0], title[0])
plt.show()
plot_conf(conf_before[1], title[1])
plt.show()

#equalised odds classifier
print("EQUALISED ODDS CLASSIFIER")
print("                 ")
print("Thresholds:")
print("African-American: %s" %tA_odds)
print("Caucasian: %s and %s" %(tC_odds[0],tC_odds[1]))
print("                 ")
print("                 ")
print("FPR and TPR:")
print("African-American")
print("FPR: %s" %FPR_TPR_odds[0][0])
print("TPR: %s" %FPR_TPR_odds[0][1])
print("Caucasian")
print("FPR: %s" %FPR_TPR_odds[1][0])
print("TPR: %s" %FPR_TPR_odds[1][1])
print("Equal opportunity classifier, African-American")
print("                 ")
print("                 ")
print("Accuracy:")
print("African-American %s" %ACC[0]) 
print("Caucasian %s" %ACC[1]) 
print("                 ")
print("Weighted accuracy: %s" %w_acc_odd)
print("                 ")
print("                 ")
print("                 ")
plot_conf(conf_odds[0], title[2])
plt.show()
plot_conf(conf_odds[1], title[3])
plt.show()
#equal opportunity
print("EQUAL OPPORTUNITY CLASSIFIER")
print("                 ")
print("Thresholds:")
print("African-American: %s" %t_odds[0])
print("Caucasian: %s" %t_odds[1])
print("                 ")
print("                 ")
print("FPR and TPR:")
print("African-American")
print("FPR: %s" %FPR_TPR_opp[0][0])
print("TPR: %s" %FPR_TPR_opp[0][1])
print("Caucasian")
print("FPR: %s" %FPR_TPR_opp[1][0])
print("TPR: %s" %FPR_TPR_opp[1][1])
print("                 ")
print("                 ")
print("Accuracy:")
print("African-American %s" %acc_after[0]) 
print("Caucasian %s" %acc_after[1])
print("                 ")
print("Weighted accuracy: %s" %w_acc_opp)
plot_conf(conf_after[0], title[4])

plot_conf(conf_after[1], title[5])




######################################################
print("NEURAL NETWORK")
print("                 ")
print("BEFORE CORRECTING FOR BIAS")
print("                 ")
print("Accuracy:")
print("African-American: %s" %acc_before_nn[0])
print("Caucasian: %s" %acc_before_nn[1])
print("                 ")
print("                 ")
print("FPR and TPR:")
print("African-American")
print("FPR: %s" %rate_before_nn[0][0])
print("TPR: %s" %rate_before_nn[0][1])
print("Caucasian")
print("FPR: %s" %rate_before_nn[1][0])
print("TPR: %s" %rate_before_nn[1][1])
print("                 ")
print("                 ")
print("                 ")
plot_conf(conf_before_nn[0], title[6])
plt.show()
plot_conf(conf_before_nn[1], title[7])
plt.show()

#equalised odds classifier
print("EQUALISED ODDS CLASSIFIER")
print("                 ")
print("Thresholds:")
print("African-American: %s" %tAodds_nn)
print("Caucasian: %s and %s" %(tCodds_nn[0],tCodds_nn[1]))
print("                 ")
print("                 ")
print("FPR and TPR:")
print("African-American")
print("FPR: %s" %FPR_TPR_odds_nn[0][0])
print("TPR: %s" %FPR_TPR_odds_nn[0][1])
print("Caucasian")
print("FPR: %s" %FPR_TPR_odds_nn[1][0])
print("TPR: %s" %FPR_TPR_odds_nn[1][1])
print("Equal opportunity classifier, African-American")
print("                 ")
print("                 ")
print("Accuracy:")
print("African-American %s" %accNN[0]) 
print("Caucasian %s" %accNN[1]) 
print("                 ")
print("Weighted accuracy: %s" %w_acc_odd_nn)
print("                 ")
print("                 ")
print("                 ")
plot_conf(conf_odds_nn[0], title[8])
plt.show()
plot_conf(conf_odds_nn[1], title[9])
plt.show()

#equal opportunity
print("EQUAL OPPORTUNITY CLASSIFIER")
print("                 ")
print("Thresholds:")
print("African-American: %s" %t_odds_nn[0])
print("Caucasian: %s" %t_odds_nn[1])
print("                 ")
print("                 ")
print("FPR and TPR:")
print("African-American")
print("FPR: %s" %FPR_TPR_opp_nn[0][0])
print("TPR: %s" %FPR_TPR_opp_nn[0][1])
print("Caucasian")
print("FPR: %s" %FPR_TPR_opp_nn[1][0])
print("TPR: %s" %FPR_TPR_opp_nn[1][1])
print("                 ")
print("                 ")
print("Accuracy:")
print("African-American %s" %acc_opp_nn[0]) 
print("Caucasian %s" %acc_opp_nn[1])
print("                 ")
print("Weighted accuracy: %s" %w_acc_opp_nn)

plot_conf(conf_opp_nn[0], title[10])
plt.show()
plot_conf(conf_opp_nn[1], title[11])
plt.show()

