# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:17:10 2020

@author: Værksted Matilde
"""

import numpy as np
import matplotlib.pyplot as plt

def percentile(p1,p2,p3):
    l1 = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    l2 = np.sqrt((p1[0]-p3[0])**2+(p1[1]-p3[1])**2)

    return l1/l2

def estimate(x,y):
    """
    Computes and returns parameters of ax+b
    x: x-koefficients of both points
    y: y-koefficients of both points    
    """
    a1 = (y[1]-y[0])/(x[1]-x[0]+0.000000000000000000000000000001)
    b1 = y[1]-a1*x[1]      
    return a1, b1

#CLVar = equal_NN
#group = 'African-American'
#p0 = [1,1] 
#T = np.arange(0,1.001,0.001)
def equal_odds(T, CLVar, group, p0, plot = False): 
    """
    Compute FPR and TPR of both groups of protected attribute (in given class)
    Find equalised odds predictor
    Plot equalised odds predictor with ROC curves
    Compute confusions matrices, FPR/TPR of equalised odds predictor. 
    Compute confusions matrices, FPR/TPR with t = 0.5
    
    Input: 
    T: List of thresholds
    CLVar: Class variable of the class "equal"
    group: The group with the lowest ROC curve
    p0: Whether to hold (0,0) or (1,1)
    
    Output: 
    FP_TP_rate_A
    FP_TP_rate_C
    ACC_A
    ACC_C
    Fpr_cau
    Tpr_cau
    Fpr_afri
    Tpr_afri
    """
    
    Fpr_rf, Tpr_rf= CLVar.ROC_(T, models = True)
    Fpr_cau, Tpr_cau = Fpr_rf['Caucasian'], Tpr_rf['Caucasian']
    Fpr_afri, Tpr_afri = Fpr_rf['African-American'], Tpr_rf['African-American']

    #accuracies of all values of t
    accs_rf =  CLVar.acc_(T, models = True)

    #Find point with highest acc on "lowest" ROC curve
    accs_rf = accs_rf[group] #all acc
    max_idx = np.argmax(accs_rf) #idx for highest acc
    maxt = T[max_idx] #t for highest acc
    maxp = [Fpr_afri[max_idx], Tpr_afri[max_idx]] #(fpr,tpr) with highest acc

    #Compute parameters of line between p0 and all points on the highest ROC curve (one at a time)
    #For all lines, estimate the y-kooordinate (TPR) of maxp. 
    yall = np.empty(len(Fpr_cau)-2)
    for i in range(len(Fpr_cau)-2):
        a1,b1 = estimate([p0[0],Fpr_cau[i+1]], [p0[1],Tpr_cau[i+1]])
        y = a1*maxp[0]+b1
        yall[i] = y

    #Compute absolute distance between the true TPR of maxp and the estimated. 
    diff = []
    for i in range(len(yall)): 
        diff.append(abs(yall[i]-maxp[1]))
    
    #Find optimal estimate of TPR of maxp
    minidx = np.argmin(diff) + 1
    p1 = [Fpr_cau[minidx], Tpr_cau[minidx]] #add 1 to get the idx that works with all
                                        #except diff and yall

    #Find percentages and thresholds of classifier that satisfies equalised odds. 
    t_afri = T[max_idx]
    
    percent = percentile(p0, maxp, p1)
    if p0 == [0,0]:
        t_cau1 = T[-1] 
    else: 
        t_cau1 = T[0]   
    t_cau2 = T[minidx]
    
    postconf_afri =  CLVar.conf_models(t_afri,0)
    postconf_cau = CLVar.calc_ConfusionMatrix(t_cau1,t_cau2, 1,percent)
    
    conf = [postconf_afri,postconf_cau]

    FPR_TPR_odd= [CLVar.FP_TP_rate(postconf_afri),CLVar.FP_TP_rate(postconf_cau)]
    
    ACC_A = CLVar.acc_with_conf(postconf_afri)
    ACC_C = CLVar.acc_with_conf(postconf_cau)
    
    ACC = [ACC_A, ACC_C]

    if plot == True: 
        
        #plot with max accu point
        plt.plot(Fpr_rf['Caucasian'], Tpr_rf['Caucasian'],'g', label = 'Caucasian')
        plt.plot(Fpr_rf['African-American'], Tpr_rf['African-American'],'b', label = 'African-american')
        plt.plot(maxp[0], maxp[1], 'bo', label = "Max accu")
        plt.plot(p0[0], p0[1], 'go')
        plt.plot(p1[0], p1[1], 'go')
        #plt.plot([p1[0],p0[0]], [p1[1],p0[1]], 'go')
        plt.plot([p0[0],p1[0]], [p0[1], p1[1]], 'r')
        plt.legend()
        plt.show()
     
    tA = t_afri
    tC = [t_cau1, t_cau2]
    return FPR_TPR_odd, ACC, conf, tA, tC



