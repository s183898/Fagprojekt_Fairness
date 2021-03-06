# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:45:39 2020

@author: Værksted Matilde
"""

from Process_data import *
from POST import *
from Process_data import A, ytrue, yhat
from Equal_opportunity import *
       
DATA = equal(A, yhat, ytrue, N=600)
#Compute confusion matrix values with the two thresholds: t1, t2, with probability p1 for group g
#t1, t2, g, p1 = 2, 7, 1, 0.9
#conf = DATA.calc_ConfusionMatrix(t1, t2, g, p1)

##Kode fra https://www.daniweb.com/programming/computer-science/tutorials/520084/understanding-roc-curves-from-scratch
T = [0,1,2,3,4,5,6,7,8,9,10]
FPR, TPR = DATA.ROC_(T, models = False)
accs = DATA.acc_(np.arange(0,11), models = False)

Atpr = TPR['African-American']
Afpr = FPR['African-American']

Ctpr = TPR['Caucasian']
Cfpr = FPR['Caucasian']

plt.plot(Cfpr,Ctpr,'g', label = 'Caucasian')
plt.plot(Afpr,Atpr,'b', label = 'African-american')
plt.plot(Cfpr[5], Ctpr[5],'go', label = "Caucasian rates t = 5")
plt.plot(Afpr[5], Atpr[5],'bo', label = "African-American rates, t = 5")
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

max_accs = [np.argmax(accs['African-American']), np.argmax(accs['Caucasian'])]

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def percentile(p1,p2,p3):
    l1 = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    l2 = np.sqrt((p1[0]-p3[0])**2+(p1[1]-p3[1])**2)
    return l1/(l2+0.000000001)



l1 = line([Cx[4],Cx[4]],[Cx[3],Cy[3]])
l2 = line([0,0],[Ax[5],Ay[5]])

inter = intersection(l1,l2)
print(inter)

perc1 = percentile([Cx[4],Cx[4]],[Cx[3],Cy[3]],[inter[0],inter[1]])
perc2 = percentile([0,0],[Ax[5],Ay[5]],[inter[0],inter[1]])

print("1. interpolation:", perc1)
print("2. interpolation:", perc2)


plt.plot(Ax,Ay, "o")
plt.plot(Cx,Cy, "o")


#plt.plot(Cx[5],Cy[5], "o")
plt.plot(Ax[5],Ay[5], "o")
plt.plot(inter[0],inter[1],"o")

c1 = DATA.calc_ConfusionMatrix(4,3, 1, perc1)
c2 = DATA.calc_ConfusionMatrix(10,5, 0, perc2)

DATA.FP_TP_rate(c1)
DATA.FP_TP_rate(c2)
plt.show()

#equalised odds
#max_acc_raw, maxt_raw, rate_raw ,conf_before_raw, conf_after_raw, acc_before_raw, acc_after_raw, rate_before_raw = equal_opportunity(0.1, T, DATA, plot = True)
