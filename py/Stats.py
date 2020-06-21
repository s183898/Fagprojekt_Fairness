import numpy as np

from sklearn.utils import resample
from Process_data import X_train, y_train, X_test, y_test, train_index, test_index, labels
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import random
from POST import *
import pandas as pd
from Equal_opportunity import equal_opportunity 
from conf_and_rates import plot_conf, rates
from randomforrest import train_test_RF
from Equalised_odds import equal_odds, estimate, percentile

#Import variables from other scripts
from Process_data import A, ytrue, yhat
from Process_data import y_train, y_test, X_train, X_test, train_index, test_index
from Permutation_test import load_classifier

A = A.values[test_index]
A = pd.DataFrame(A)
ytrue = ytrue.values[test_index]
ytrue = pd.DataFrame(ytrue)

"""
model_nn = load_classifier("NN")
y_nn = model_nn.predict(X_test)

#Compute FPR and TPR of NN. And define class variable of equal class. 
equal_NN = equal(A[0],y_nn,ytrue[0], N =400)

# Equal odds NN

# Cauc

t1, t2, g, p2 = 0, 0.824, 1, 0.8694443684906432 

conf = equal_NN.calc_ConfusionMatrix(t1, t2, g, p2)

acc_cauc_odd = equal_NN.acc_with_conf(conf)

# Afr

t2 = 0.632

conf_afr_odds = equal_NN.conf_models(t2, 0)

acc_afr_odd = equal_NN.acc_with_conf(conf_afr_odds)


# Equal opp

# Cauc

conf_cauc_opp = equal_NN.conf_models(0.758, 1)

acc_cauc_opp = equal_NN.acc_with_conf(conf_cauc_opp)

# Afr

conf_afr_opp = equal_NN.conf_models(0.579, 0)

acc_NN_afr = equal_NN.acc_with_conf(conf_afr_opp)


print(acc_NN_afr)

"""


def permutation(n_perm, name, plots = False):
    # load model and pick attributes to permutate
    model = load_classifier(name)    
    n_attr = X_test.shape[1]

    # Initalise empty matrix for accuracies
    accs = np.zeros([n_perm,n_attr])

    for idx in range(n_attr):
        for trial in range(n_perm):
            # Define a new dataframe independent of X_test and permutate
            
            no_protected = X_test[:,[11,13]] == 0

            un_bin = (np.logical_and(no_protected[:,0],no_protected[:,1]))
            X_perm = np.array(X_test[un_bin])
            y_un = np.array(y_test[un_bin])

            X_perm[:,idx] = resample(X_perm[:,idx] ,replace = False)
            
            # make predictions
            accuracy = model.evaluate(X_perm,y_un)[1]
            
            # Compute FPR and TPR of NN. And define class variable of equal class. 
            # equal_NN = equal(A[0],y_nn,ytrue[0], N = 400)

            # conf_afr_opp = equal_NN.conf_models(0.579, 0)

            # acc_afr_opp = equal_NN.acc_with_conf(conf_afr_opp)
   

            # Append model accuracy under permutation
            accs[trial,idx] = accuracy

    if plots == True:
        # Plot mean of accuracy for each attribute
        accs_mean = np.mean(accs, axis = 0)
        # Errors deminish when the number of permutations are 1000, so these are ignored
        error_accs = np.std(accs, axis = 0 )/np.sqrt(n_perm)
        
        y_pos = np.arange(n_attr)
        plt.barh(y_pos,accs_mean)
        plt.title(name + " - " + "Accuracy")
        plt.yticks(y_pos,labels)
        plt.show()

    return accs


# no_protected = X_test[:,[11,13]] == 0

# un_idx = (np.logical_and(no_protected[:,0],no_protected[:,1]))

# X_un = X_test[un_idx]

#accs = permutation(1000,"NN")

#np.save("accs_unprotected", accs)




n_a = sum(X_test[:,11] == 1)
n_c = sum(X_test[:,13] == 1)
n_all = len(y_test) - n_a - n_c

"""
print(n_all)

accs_no = np.load("accs_unprotected.npy")
accs_afr = np.load("accs_afr_odd.npy")
accs_cauc = np.load("accs_cauc_odd.npy")

print(np.shape(accs_afr))

accs_all = (accs_no*n_all+accs_afr*n_a+accs_cauc*n_c)/len(y_test)

accs_all[:,13] = accs_cauc[:,13]
accs_all[:,11] = accs_afr[:,11]

print(accs_all.shape)

"""
accs = []

model = load_classifier("NN")

y_nn = model.predict(X_test)

t1, t2, g, p2 = 0, 0.824, 1, 0.8694443684906432 

equal_NN = equal(A[0],y_nn,ytrue[0], N = 400)

conf = equal_NN.calc_ConfusionMatrix(t1, t2, g, p2)

acc_cauc_odd = equal_NN.acc_with_conf(conf)

###################

t2 = 0.632

conf_afr_odds = equal_NN.conf_models(t2, 0)

acc_afr_odd = equal_NN.acc_with_conf(conf_afr_odds)

accs_no = 0.8430

acc_compare = (acc_afr_odd*n_a+acc_cauc_odd*n_c+accs_no*n_all)/len(y_test)

print(acc_compare)

np.save("acc_odds_null", acc_compare)