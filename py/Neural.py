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

"""
# Attributes
X = twoyears.data.drop(['decile_score.1'],axis = 1).values

# binary labels
y = twoyears.data["decile_score.1"].values

# train and test split
upper = int(np.floor(6907*0.8))

X_train = X[0:upper]
X_test = X[upper:]
y_train = y[0:upper]
y_test = y[upper:]
"""

model = load_model("./NN_model.h5")

def train_NN(n_epoch):
       model = Sequential()
       model.add(Dense(29, activation='relu'))
       model.add(Dense(32, activation='relu'))
       model.add(Dropout(0.2))
       model.add(Dense(16, activation='relu'))
       model.add(Dense(1, activation='sigmoid'))

       model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

       model.fit(X_train, y_train, epochs=n_epoch, batch_size=10)

       model.save("C:/Users/rasmu/OneDrive/Dokumenter/4. semester/Fagprojekt/02466---Project-in-Artificial-Intelligence-and-Data-Science/py/NN_model.h5")
       
       return model

#model = train_NN(150)

_, accuracy = model.evaluate(X_test, y_test)

print('Accuracy: %.2f' % (accuracy*100))


"""
def ROC_NN(A):

       A = A.values[test_index]
       A = pd.DataFrame(A)

       pred = model.predict(X_test)

       equal_NN = equal(A[0],pred,ytrue.values[test_index], N =400)

       T = np.arange(0,1.001,0.001)

       FPR, TPR = equal_NN.ROC_(T, models = True)

       return T , equal_NN, FPR, TPR

T, equal_NN, FPR, TPR = ROC_NN(A)

accs_NN = equal_NN.acc_(T, models = True)

lower_ROC = accs_NN["African-American"]

max_idx = np.argmax(lower_ROC)

best_thres = T[max_idx]

FPRA = FPR["African-American"]
TRPA = TPR["African-American"]

FPRC = FPR["Caucasian"]
TPRC = TPR["Caucasian"]

max_A = [FPRA[max_idx], TRPA[max_idx]]

def estimate(x,y):
        a1 = (y[1]-y[0])/(x[1]-x[0]+0.000000000000000000000000000001)
        b1 = y[1]-a1*x[1]      
        return a1, b1

yall = np.empty(len(FPRC)-2)

p0 = [1,1]

for i in range(len(FPRC)-2):
    a1,b1 = estimate([p0[0],FPRC[i+1]], [p0[1],TPRC[i+1]])
    y = a1*max_A[0]+b1
    yall[i] = y

diff = []
for i in range(len(yall)): 
    diff.append(abs(yall[i]-max_A[1]))

minidx = np.argmin(diff) + 1

max_C = [FPRC[minidx],TPRC[minidx]]

perc = percentile(p0,max_A,max_C)
if p0 == [0,0]:
    t_cau1 = T[-1] 
else: 
    t_cau1 = T[0]

Conf_A =  equal_NN.conf_models(best_thres,0)
Conf_C = equal_NN.calc_ConfusionMatrix(t_cau1,T[minidx],1,perc)

print(equal_NN.FP_TP_rate(Conf_A))
print(equal_NN.FP_TP_rate(Conf_C))

plt.plot(max_A[0],max_A[1], "o")
plt.plot(max_C[0],max_C[1], "o")

plt.plot(FPR['Caucasian'], TPR['Caucasian'], label = 'caucasian')
plt.plot(FPR['African-American'], TPR['African-American'], label = 'african-american')

plt.legend()
plt.show()

"""
"""

labels = twoyears.data.drop(["decile_score.1"],axis =1)
labels = labels.columns.values


def Perm_NN():
       X_perm = X_test

       trial_n = 10
       feature_n = 29

       accs = np.zeros([trial_n,feature_n])
       losses = np.zeros([trial_n,feature_n])

       for idx in range(feature_n):
              for trial in range(trial_n):
                     X_perm = X_test
                     X_perm[:,idx] = resample(X_perm[:,idx] ,replace = False)
                     perm_loss, perm_accuracy = model.evaluate(X_perm, y_test)
                     accs[trial,idx] = perm_accuracy 
                     losses[trial,idx] = perm_loss

       loss_mean = np.mean(losses,axis=0)
       accs_mean = np.mean(accs, axis = 0)

       error_loss = np.std(losses, axis = 0)/np.sqrt(trial_n)
       error_accs = np.std(accs, axis = 0 )/np.sqrt(trial_n)

       y_pos = np.arange(len(labels))

       plt.subplot(211)

       plt.barh(y_pos, loss_mean, xerr = error_loss)
       plt.title("Loss")
       plt.yticks(y_pos,labels)

       plt.subplot(212)

       plt.barh(y_pos,accs_mean)
       plt.title("Accuracy")
       plt.yticks(y_pos,labels)
       plt.show()

       return losses, accs
"""