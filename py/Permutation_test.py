import numpy as np
from sklearn.utils import resample
from Process_data import X_train, y_train, X_test, y_test, train_index, test_index, labels
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
from sklearn.ensemble import RandomForestClassifier
from scipy import stats


np.random.seed(217)
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
    model = load_classifier(name)

    X_perm = X_test
    n_attr = X_test.shape[1]

    accs = np.zeros([n_perm,n_attr])
    losses = np.zeros([n_perm,n_attr])

    for idx in range(n_attr):
        for trial in range(n_perm):
            X_perm = X_test
            X_perm[:,idx] = resample(X_perm[:,idx] ,replace = False)
            if name == "NN":
                perm_loss, perm_accuracy = model.evaluate(X_perm, y_test, verbose = 0)
            elif name == "RF":
                perm_accuracy = model.score(X_perm,y_test)
            else:
                print("Wrong name")
                return

            accs[trial,idx] = perm_accuracy 
            #losses[trial,idx] = perm_loss
                     
    #loss_mean = np.mean(losses,axis=0) 
    accs_mean = np.mean(accs, axis = 0)

    #error_loss = np.std(losses, axis = 0)/np.sqrt(n_perm)
    error_accs = np.std(accs, axis = 0 )/np.sqrt(n_perm)

    y_pos = np.arange(n_attr)

#    plt.subplot(211)

#    plt.barh(y_pos, loss_mean, xerr = error_loss)
#    plt.title("Loss")
#    plt.yticks(y_pos,labels)

#    plt.subplot(212)
    if plots == True:
        plt.barh(y_pos,accs_mean)
        plt.title(name + " - " + "Accuracy")
        plt.yticks(y_pos,labels)
        plt.show()

    return accs

def permutation_test(n_perm, name):

    accs = permutation(n_perm, name)
    
    p_values = []

    for idx in range(accs.shape[1]):
        p = stats.ttest_1samp(accs[:,idx],0.81)[1]
        p_values.append(p)

    return p_values


#print(permutation(1,"NN"))

p_values = permutation_test(2, "RF")

#np.save("p_values_29", p_values)




