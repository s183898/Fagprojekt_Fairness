import numpy as np
from sklearn.utils import resample
from Process_data import X_train, y_train, X_test, y_test, train_index, test_index, labels
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import random



#np.random.seed(217)
def load_classifier(name):

    if name == "NN":

        model = load_model("./NN_model.h5")

    elif name == "RF":
        model = RandomForestClassifier(n_estimators=100,
                               criterion = 'entropy',
                               min_samples_split=2,
                               bootstrap = True,
                               max_features = None)
        
        model.fit(X_train, y_train)
        
    else:
        print("Wrong model name")

    return model

def permutation(n_perm, name, plots = False):
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

    # Access accuracy of permutations with a specifik model 
    accs = permutation(n_perm, name)
    p_values = []

    for idx in range(accs.shape[1]):
        # Calculate p-value from t-test for each feature based on permutations
        p = stats.ttest_1samp(accs[:,idx],0.81)[1]
        p_values.append(p)

    return p_values

def direction(n_perm, name, full = True, plots = False):
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
            print(bin_1)
            # Predict based on permutations
            pred_perm = model.predict_proba(X_perm)[:,0]
            print(pred_perm)
            # Save mean of predictions
            mean_score = sum(pred_perm[bin_1])/sum(bin_1)
            decile[trial,idx] = mean_score

    if plots == True:
        mean_decile = np.mean(decile,axis = 0)

        y_pos = np.arange(n_attr)
        plt.barh(y_pos,mean_decile)
        plt.title(name + " - " + "Influence")
        plt.yticks(y_pos,labels[binary:])
        plt.show()

    return decile

#print(direction(1,"RF", plots = True))
#print(direction(5,"NN", plots = True))

for i, lab in enumerate(labels):
    print(sum(X_train[:,i]))
    print(sum(X_test[:,i]))
    print(lab)
    print(i)

