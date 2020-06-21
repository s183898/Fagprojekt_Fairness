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

# A = A.values[test_index]
# A = pd.DataFrame(A)
# ytrue = ytrue.values[test_index]
# ytrue = pd.DataFrame(ytrue)

# A = A.values
# A = pd.DataFrame(A)
# ytrue = ytrue.values
# ytrue = pd.DataFrame(ytrue)


# equal_NN = equal(A[0],yhat,ytrue[0], N = 400)
# conf = equal_NN.conf_models(5, 1)
# print(equal_NN.FP_TP_rate(conf))


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
    decile = np.zeros([n_perm,n_attr])

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
            mean_score = sum(pred_perm[bin_1])/sum(bin_1)
            decile[trial,idx] = mean_score
            mean_decile = np.mean(decile,axis = 0)
            mean_decile = abs(mean_decile-1)*10


    if plots == True:
    
        y_pos = np.arange(n_attr)
        plt.barh(y_pos,mean_decile)
        plt.title(name + " - " + "mean decile score after permutations")
        plt.yticks(y_pos,labels[binary:])
        plt.xlabel("Decile score")
        plt.show()

    return mean_decile


p_values = permutation_test(1000,"RF")

np.save("npy_files/p_value_RF_CV_29",p_values)

print(np.load("npy_files/p_value_RF_CV_29.npy"))