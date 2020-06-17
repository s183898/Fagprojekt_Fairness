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



np.random.seed(217)

def train_NN(x):
    model = Sequential()
    model.add(Dense(29, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=150, batch_size=10)

    loss, accuracy = model.evaluate(X_test,X_train)

    return accuracy

_, accuracy = model.evaluate(X_test, y_test)

print('Accuracy: %.2f' % (accuracy*100))

