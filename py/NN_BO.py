import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.utils import resample
from Process_data import twoyears
import numpy as np
import matplotlib.pyplot as plt
from POST import equal
from Process_data import A, ytrue, yhat
from Process_data import X_train, y_train, X_test, y_test, train_index, test_index
from Post_RawData import line, percentile
import GPyOpt
import matplotlib.pyplot as plt

np.random.seed(217)

train_idx = int(np.floor(X_train.shape[0]*0.8))

X_train2 = np.array(X_train[:train_idx,:])
X_test2 = np.array(X_train[train_idx:,:])

y_train2 = np.array(y_train[:train_idx])
y_test2 = np.array(y_train[train_idx:])

print(X_train2.shape)
print(X_test2.shape)
print(y_train2.shape)
print(y_test2.shape)


# Build neural network classifier with exchangable hyper-parameters

def train_NN(x):

    parameters = x[0]

    model = Sequential()
    model.add(Dense(29, activation='relu'))
    model.add(Dense(int(parameters[2]), activation='relu'))
    model.add(Dropout(parameters[0]))
    model.add(Dense(int(parameters[3]), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer_dict[parameters[5]], metrics=['accuracy'])

    model.fit(X_train2, y_train2, epochs= int(parameters[1]),  batch_size= int(parameters[4]))

    loss, accuracy = model.evaluate(X_test2,y_test2)

    return -accuracy


# define constraints for hyper-parameters

# dropout rate
dropout = np.arange(0,0.4, 0.05)

# number of epochs 
n_epochs = np.arange(100,200,50)

# Hidden neurons and batch size
hidden_1 = [2**x for x in range(3,7)]
hidden_2 = [2**x for x in range(1,7)]
batch = [2**x for x in range(3,7)]

# optimizer (SGD, Adam)
optimizer_dict = {0: "adam", 1: "sgd"}
optimizer = (0, 1)

domain = [  {'name': 'dropout', 'type': 'discrete', 'domain': dropout},
            {'name': 'epochs', 'type': 'discrete', 'domain': n_epochs},
            {'name': 'hidden_1', 'type': 'discrete','domain': hidden_1},
            {'name': 'hidden_2', 'type': 'discrete','domain': hidden_2},
            {'name': 'batch', 'type': 'discrete','domain': batch},
            {'name': 'optimizer', 'type': 'categorical', 'domain': optimizer}]


# run optimisation 

opt = GPyOpt.methods.BayesianOptimization(f = train_NN,  
                                              domain = domain,        
                                              acquisition_type = 'EI' ,
                                             )



opt.run_optimization(max_iter=15)

opt.plot_acquisition()

# extract hyper-parameters that maximises accuracy
x_best = opt.X[np.argmin(opt.Y)]

print(f"Best accuracy was obtained at {opt.fx_opt*-1} %")
print("The best parameters obtained:")
print("dropout rate = " + str(x_best[0]) + ", number of epochs = " + str(x_best[1]) + ", neruons in layer 1 = " + str(x_best[2]) + ", neurons in layer 2 =" + str(x_best[3]) + ", batch size = " + str(x_best[4]) + ", optimizer = " + str(optimizer_dict[x_best[5]]))


np.save("NN_BO_2.h5", x_best)

