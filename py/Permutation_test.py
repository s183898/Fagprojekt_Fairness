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
from Process_data import A, ytrue, yhat


A = A.values
A = pd.DataFrame(A)
ytrue = ytrue.values
ytrue = pd.DataFrame(ytrue)
yhat = yhat.values
yhat = pd.DataFrame(yhat)


p_NN = np.load("npy_files\p_value_NN_BO.npy")

def permutation_decile(n_perm):

    equal_true = equal(A[0], yhat[0], ytrue[0], N=400)
    conf_0_true = equal_true.conf_(5,0)
    conf_1_true = equal_true.conf_(5,1)
    FP_0_true, TP_0_true = equal_true.FP_TP_rate(conf_0_true)
    FP_1_true, TP_1_true = equal_true.FP_TP_rate(conf_1_true)

    print(FP_0_true, TP_0_true)
    print(FP_1_true, TP_1_true)
    TP_list_0 = []
    TP_list_1 = []
    FP_list_0 = []
    FP_list_1 = []
    for i in range(n_perm):
        yperm = np.array(resample(yhat[0]))
        equal_perm = equal(A[0], yperm, ytrue[0], N=400)
        conf_0 = equal_perm.conf_(5,0)
        conf_1 = equal_perm.conf_(5,1)
        FP_0, TP_0 = equal_perm.FP_TP_rate(conf_0)
        FP_1, TP_1 = equal_perm.FP_TP_rate(conf_1)

        FP_list_0.append(FP_0)
        FP_list_1.append(FP_1)
        TP_list_0.append(TP_0)
        TP_list_1.append(TP_1)

    p_FP_0 = stats.ttest_1samp(FP_list_0,)
    p_FP_1 = stats.ttest_1samp(FP_list_0,)
    p_FP_1 = stats.ttest_1samp(FP_list_0,)
    p_FP_1 = stats.ttest_1samp(FP_list_0,)

    return 



#np.random.seed(217)
def load_classifier(name):
    """ Loads best classifier with given initials """

    if name == "NN":

        model = load_model("NN_model_with_BO_finished.h5")

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

def permutation(n_perm, name, plots = False):
    """" Makes permutations to test if a feature has a significant effect on accuracy"""

    # load model and pick attributes to permutate
    model = load_classifier(name)    
    n_attr = X_test.shape[1]

    # Initalise empty matrix for accuracies
    accs = np.zeros([n_perm,n_attr])

    for idx in range(n_attr):
        for trial in range(n_perm):
            # Define a new dataframe independent of X_test and permutate
            X_perm = np.array(X_test)
            X_perm[:,idx] = resample(X_perm[:,idx] ,replace = False)
            # Select model and calculate accuracies under permutations

            if name == "NN":    
                perm_loss, perm_accuracy = model.evaluate(X_perm, y_test, verbose = 0)
            elif name == "RF":
                perm_accuracy = model.score(X_perm,y_test)
            else:
                print("Wrong name")
                return
            # Append model accuracy under permutation
            accs[trial,idx] = perm_accuracy 

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

def permutation_test(n_perm, name):
    """ recieves a list of accuracies and tests if these are significantly different than the original accuracy """

    # Access accuracy of permutations with a specifik model 
    accs = permutation(n_perm, name)
    p_values = []

    if name == "NN":
        _, accs_ori = load_classifier(name).evaluate(X_test,y_test)
    elif name == "RF":
        accs_ori = load_classifier(name).score(X_test,y_test)

    for idx in range(29):
        # Calculate p-value from t-test for each feature based on permutations
        p = stats.ttest_1samp(accs[:,idx],accs_ori)[1]
        p_values.append(p)

    return p_values

def direction(n_perm, name, full = False, plots = False):
    """ Makes permutations and saves average decile score,
    to see which direction an attribute influences the prediction """

    model = load_classifier(name)
    # Index of first binary feature
    binary = 6
    n_attr = (X_test.shape[1]-binary)
    prob = np.zeros([n_perm,n_attr])

    for idx in range(n_attr):
        i = idx + binary
        for trial in range(n_perm):
            # Define a new dataframe independent of X_test and permutate
            X_perm = np.array(X_test)
            if full:
                X_perm = resample(X_perm)
            else:
                X_perm[:,i] = resample(X_perm[:,i], replace = False)
            # Grab values where features are 1 / True
            bin_1 = X_perm[:,i] == True
            # Predict based on permutations
            if name == "NN":
                pred_perm = model.predict(X_perm)
            if name == "RF":
                pred_perm = model.predict_proba(X_perm)[:,1]
            # Save mean of predictions


            mean_score = np.mean(pred_perm[bin_1])
            prob[trial,idx] = mean_score
            mean_prob = np.mean(prob,axis = 0)
            mean_decile = abs(mean_prob-1)*10


    if plots == True:
        if name == "NN":
            ori_prob = np.mean(model.predict(X_test))
            ori_decile = abs(ori_prob-1)*10
        if name == "RF":
            ori_prob = np.mean(model.predict_proba(X_test)[:,1])
            ori_decile = abs(ori_prob-1)*10

        fig = plt.figure()
        errs = np.std(prob, axis = 0)*10

        y_pos = np.arange(n_attr)
        colors = []
        for i in mean_decile:
            if i > ori_decile:
                colors.append("r")
            elif i < ori_decile:
                colors.append("g")
            else:
                colors.append("b")

        plt.barh(y_pos,mean_decile, xerr =[errs,errs], color = colors)
        plt.axvline(ori_decile, linestyle = ":", color = "r")
        plt.title(name + " - " + "mean decile score after permutations")
        plt.yticks(y_pos,labels[binary:])
        plt.xlabel("Decile score")
        plt.tight_layout()
        plt.show()
        #fig.savefig('direction_NN_1000_full.png')

    return mean_decile

direction(1000, "NN",full = False,plots = True)

