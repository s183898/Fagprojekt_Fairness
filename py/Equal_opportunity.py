# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:13:54 2020

@author: Værksted Matilde
"""

#PR_A1, TPR_C1 = Tpr_afri, Tpr_cau
import matplotlib.pyplot as plt
import numpy as np
#from POST import *

#sigma, T, CLVar= sigma, T, Equal_rf
def equal_opportunity(sigma, T, CLVar, plot = False):
    """
    Equal_opportunity computes conf matrx, acc, (TPR, FPR) before and after the equal opportunity step is aplied
    
    
    T: list of thresholds
    sigma: Accepted difference between  TPR_A and TPR_C
    CLVar: Name of equal class variable 
    
    max_acc: Maximal weighted accuracy 
    maxtA: maximal threshold african american
    maxtC: Maximal threshold caucasian
    rateA: FPR and TPR african american
    rateC: FPR and TPR caucasian
    conf_before_A: conf matrix (both groups) with threshold 0.5
    conf_after_C: conf mtrx (both groups) of equal opportunity classifier
    """

    #Compute FPR and TPR 
    FPR, TPR = CLVar.ROC_(T, models = True)
    FPR_C, TPR_C1 = FPR['Caucasian'], TPR['Caucasian']
    FPR_A, TPR_A1 = FPR['African-American'], TPR['African-American']

    TPR_A = np.asarray(TPR_A1)
    TPR_C = np.asarray(TPR_C1)
    #Define 2 thresholds
    T_A = np.asarray(T)
    T_C = np.asarray(T)
    
    #remove i's from TPR of both races, and remove coherent thresholds
    #for i in range(2):
        
    #    idxa = TPR_A != i
    #    idxc = TPR_C != i
        
    #    TPR_A = TPR_A[idxa]
    #    TPR_C = TPR_C[idxc]
   # 
   #     T_A = T_A[idxa]
   #     T_C = T_C[idxc]
    
    TtotalC = []
    TtotalA = []
    for t_idx, A in enumerate(TPR_A): 
        
        #Create list with all entries = A and the same length as TPR_C
        A =[A]
        A = len(TPR_C)*A
        
        #Find difference between A and all TPR of C
        Diff = np.asarray(A) - np.asarray(TPR_C)
        
        #Define idx of TPR_C that are identical to A and store them in list 
        idx = abs(Diff) < sigma
        if sum(idx) > 0:    
            t = T_C[idx]
            for i in range(len(t)):
                TtotalC.append(t[i])
                TtotalA.append(T_A[t_idx])

            
    #compute (TPR, FPR) and accuracy 
    pairsA, pairsC = [], []
    accA, accC = [], []
    for a,c in zip(TtotalA, TtotalC):
        
        confa = CLVar.conf_models(a,0) #conf mtrx with threshold a, for african-american
        confc = CLVar.conf_models(c,1) #conf mtrx with threshold c for caucasian
        
        pairsA.append(CLVar.FP_TP_rate(confa)) #collect (FPR, TPR) with threshold i, for african-american
        pairsC.append(CLVar.FP_TP_rate(confc))  #collect (FPR, TPR) with threshold i for caucasian
        
        accA.append(CLVar.acc_with_conf(confa)) #collect accuracy with threshold i, for african-american
        accC.append(CLVar.acc_with_conf(confc)) #collect accuracy with threshold i for caucasian
    
    
    #compute highest accuracy 
    N_A = CLVar.Freq[0] #number of 
    N_C = CLVar.Freq[1]
    
    weighted_acc = []
    for a, c in zip(accA, accC):
        weighted_acc.append((a*N_A+c*N_C))
    
        
    #maximum accuracy of pairs
    maxw = np.argmax(weighted_acc)
    max_acc = weighted_acc[maxw]
    maxt = [TtotalA[maxw], TtotalC[maxw]]
    rate = [pairsA[maxw], pairsC[maxw]]
    
    #Conf mtrx before 
    conf_before_A = CLVar.conf_models(0.5, 0)
    conf_before_C = CLVar.conf_models(0.5, 1)
    conf_before = [conf_before_A,conf_before_C]
    
    #conf mtrx after
    conf_after_A = CLVar.conf_models(maxt[0], 0)
    conf_after_C = CLVar.conf_models(maxt[1], 1)
    conf_after = [conf_after_A,conf_after_C]
    
    #acc 
    acc_before = [CLVar.acc_with_conf(conf_before_A), CLVar.acc_with_conf(conf_before_C)]
    acc_after = [CLVar.acc_with_conf(conf_after_A), CLVar.acc_with_conf(conf_after_C)]
    
    #(FPR, TPR) before
    rate_before = [CLVar.FP_TP_rate(conf_before_A), CLVar.FP_TP_rate(conf_before_C)]
    
    
    if plot == True: 
        
        #plot with max accu point
        plt.plot(FPR_C, TPR_C1,'g', label = 'Caucasian')
        plt.plot(FPR_A, TPR_A1,'b', label = 'African-american')
        plt.plot(rate[0][0],rate[0][1] ,'b*', label = "Rates with optimal threshold")
        plt.plot(rate[1][0],rate[1][1], 'g*', label = "Rates with optimal threshold")
        plt.plot([rate[0][0],rate[1][0]],[rate[0][1],rate[1][1]] ,'r')
        plt.legend()
        plt.title("Equal opportunity")
        plt.show() 
    
 
    return max_acc, maxt, rate ,conf_before, conf_after, acc_before, acc_after, rate_before

#TPR_A1, TPR_C1, FPR_A, FPR_C, sigma, T, CLVar = Tpr_afri, Tpr_cau, Fpr_afri, Fpr_cau, Sigma, T, Equal_rf
