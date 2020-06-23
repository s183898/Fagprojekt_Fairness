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
from conf_and_rates import plot_conf, rates
import numpy as np
from randomforrest import train_test_RF
from tensorflow.keras.models import load_model
import pickle
from randomforrest import train_test_RF, load_classifier_1

from Equalised_odds import equal_odds, estimate, percentile

#Import variables from other scripts
from Process_data import A, ytrue, yhat
from Process_data import y_train, y_test, X_train, X_test, train_index, test_index


np.random.seed(217)
#Prepair A anf ytrue for models
A = A.values[test_index]
A = pd.DataFrame(A)
ytrue = ytrue.values[test_index]
ytrue = pd.DataFrame(ytrue)
yhat = yhat.values[test_index]
yhat = pd.DataFrame(yhat)

#thresholds and sigma
T = np.arange(0,1.001,0.001)
sigma = 0.001

#%% Random forrest
#model_rf = load_classifier_1("RF", X_train, y_train)
train_acc, test_acc, yhat_rf, model_rf = train_test_RF(X_train, y_train, X_test, y_test)
print("Training accuracy, RF: %s" %train_acc)
print("Test accuracy, RF:     %s" %test_acc)


#Define equal class variable with two_year_recid[test_index], race[test_index] and predictions on test set
Equal_rf = equal(A[0], yhat_rf[0], ytrue[0], N=400)

#Equalised odds
group = 'African-American'
p0 = [1,1]
FPR_TPR_odds, ACC, conf_odds, tA_odds, tC_odds, percent_RF= equal_odds(T, Equal_rf, group, p0, plot = True)

#Equal opportunity
max_acc, t_odds, FPR_TPR_opp,conf_before, conf_after, acc_before, acc_after, rate_before= equal_opportunity(sigma, T, Equal_rf, plot = True)
 

#Define number of observations in each class
n_A = Equal_rf.Freq[0]
n_C = Equal_rf.Freq[1]

#weighted relative acc
w_acc_odd = (ACC[0]*n_A + ACC[1]*n_C)/(n_A+n_C)
w_acc_opp = (acc_after[0]*n_A + acc_after[1]*n_C)/(n_A+n_C)
w_acc_before = (acc_before[0]*n_A + acc_before[1]*n_C)/(n_A+n_C)

##Rates 
#PPV, TDR, FOR, FNR, FDR, FPR, NPV, TNR
Rates_rf_before_A = rates(conf_before[0])
Rates_rf_odds_A = rates(conf_odds[0])
Rates_rf_opp_A = rates(conf_after[0])
Rates_rf_before_C = rates(conf_before[1])
Rates_rf_odds_C = rates(conf_odds[1])
Rates_rf_opp_C = rates(conf_after[1])

Baserate = 

def out(rate):     
    print("& %.3f & %.3f & %.3f & %.3f & %.3f & %.3f " %(rate[0], rate[1], rate[2], rate[3], rate[4], rate[5]))
    
#out(Rates_rf_odds_C )

#%% NN

model_nn = load_classifier_1("NN",X_train, y_train)
y_nn = model_nn.predict(X_test)
loss_nn_test, acc_nn_test = model_nn.evaluate(X_test, y_test, verbose = 0)
loss_nn_train, acc_nn_train = model_nn.evaluate(X_train, y_train, verbose = 0)

print("Test accuracy, NN:     %s" %acc_nn_test)
print("Test loss, NN:     %s" %loss_nn_test)
print("Training accuracy, NN: %s" %acc_nn_train)
print("Training loss, NN: %s" %loss_nn_train)


#Compute FPR and TPR of NN. And define class variable of equal class. 
equal_NN = equal(A[0],y_nn,ytrue[0], N =400)

n_A = equal_NN.Freq[0]
n_C = equal_NN.Freq[1]
#Equalised odds
group = 'African-American'
p0 = [1,1]
FPR_TPR_odds_nn, accNN, conf_odds_nn, tAodds_nn, tCodds_nn, percent_NN = equal_odds(T, equal_NN, group, p0, plot = True)

#Equal opportunity
max_acc_nn, t_odds_nn, FPR_TPR_opp_nn,conf_before_nn, conf_opp_nn, acc_before_nn, acc_opp_nn, rate_before_nn = equal_opportunity(sigma, T, equal_NN, plot = True)

#weighted relative acc
w_acc_odd_nn = (accNN[0]*n_A + accNN[1]*n_C)/(n_A+n_C)
w_acc_opp_nn = (acc_opp_nn[0]*n_A + acc_opp_nn[1]*n_C)/(n_A+n_C)
w_before_nn = (acc_before_nn[0]*n_A + acc_before_nn[1]*n_C)/(n_A+n_C)

Rates_nn_before_A = rates(conf_before_nn[0])
Rates_nn_odds_A = rates(conf_odds_nn[0])
Rates_nn_opp_A = rates(conf_opp_nn[0])
Rates_nn_before_C = rates(conf_before_nn[1])
Rates_nn_odds_C = rates(conf_odds_nn[1])
Rates_nn_opp_C = rates(conf_opp_nn[1])

out(Rates_nn_opp_A )

#%%
#Raw data
from Process_data import A, ytrue, yhat
A = pd.DataFrame(A.values)
DATA = equal(A[0], yhat, ytrue, N=600)
T = [0,1,2,3,4,5,6,7,8,9,10]
FPR, TPR = DATA.ROC_(T, models = False)
accs = DATA.acc_(np.arange(0,11), models = False)

Atpr = TPR['African-American']
Afpr = FPR['African-American']

Ctpr = TPR['Caucasian']
Cfpr = FPR['Caucasian']

plt.plot(Cfpr,Ctpr,'g', label = 'Caucasian')
plt.plot(Afpr,Atpr,'b', label = 'African-american')
plt.plot(Cfpr[5], Ctpr[5],'go', label = "Rates, Caucasian, t = 5")
plt.plot(Afpr[5], Atpr[5],'bo', label = "Rates, African-American, t = 5")
plt.legend()
plt.title("ROC curve on decile_score.1 and two_years_recid")
plt.show() 

accs = DATA.acc_(np.arange(0,11), models = False)

print(accs['African-American'][5])
print(accs['Caucasian'][5])
print(Cfpr[5])
print(Ctpr[5])
print(Afpr[5])
print(Atpr[5])
conf_A = DATA.conf_(5,0)
conf_C = DATA.conf_(5,1)
print(rates(conf_A))
print(rates(conf_C))

plot_conf(conf_A, 'African-American')
plot_conf(conf_C, 'Caucasian')
#%% Print results

title = ["RF classifier (African-American)", "RF classifier (Caucasian)", "RF classifier (African-American)", "RF classifier (Caucasian)", "RF classifier (African-American)","RF classifier (Caucasian)",
         "NN, classifier (African-American)", "NN classifier (Caucasian)", "NN, classifier (African-American)", "NN classifier (Caucasian)", "NN, classifier (African-American)","NN classifier (Caucasian)"]

print("RANDOM FORREST")
print("                 ")
print("BEFORE CORRECTING FOR BIAS")
print("                 ")
print("Accuracy:")
print("African-American: %s" %acc_before[0])
print("Caucasian: %s" %acc_before[1])
print("weighted: %s" %w_acc_before)
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
print("Weighted accuracy: %s" %w_before_nn)
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
print("Percent (caucasian): %s" %percent_NN)
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

