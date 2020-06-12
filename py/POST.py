# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:05:59 2020

@author: Bruger
"""

#from Process_data import *
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib as plt
import matplotlib.pyplot as plt
import collections
ConfusionMatrix = collections.namedtuple('conf', ['tp','fp','tn','fn']) 
#from Process_data import A, ytrue, yhat


class equal:
    def __init__(self,A, yhat, ytrue, N):
        """ data_path: path to csv file
            yhatName: Name of attribute containing yhat
            ytrueName: Name of attribute containing ytrue
            AName: Name of protected attribute
            N: Read describtion of "Groups" below
        """
        
        #self.data = data
        #Y = self.data[[yhatName, ytrueName]].to_numpy(dtype=float)
        self.A = A.to_numpy()
        self.Yhat = yhat
        self.Ytrue = ytrue
        
        #Groups: List with dictionaries. Each group of A, where number of observations > N has a dictionary
        #Each dictionary has 3 keys: ytrue, yhat and A
        Groups = []
        Race, Freq = np.unique(self.A,return_counts=True)
        self.Race = Race[Freq>N]
        self.Freq = Freq[Freq>N]
        #self.Race = Race
        for i in range(len(self.Race)):
            Groups.append({})
            Groups[i]['groupname'] = self.Race[i] 
            Groups[i]['ytrue'] = self.Ytrue[self.A == self.Race[i]]
            Groups[i]['yhat'] = self.Yhat[self.A == self.Race[i]]
            
        self.Groups = Groups

    
    def calc_ConfusionMatrix(self,t1, t2, g, p2, OnlyOne = False, positive_label = 1):
        """
        Computes confusionsmatrix for randomised predictor
        
        Computestp, fp, tn, fn for all groups by default.
        If OnlyOne is True it only computes for one group
        
        p2: probability of t2, which is the distance from t1 to the "correct point"
        "t1 and t2": thresholds
        
        """
        if t1 < t2: 
            l = [1,0]
        elif t1 > t2: 
            l = [0,1]
        tp=fp=tn=fn=0
        bool_actuals = [act==positive_label for act in self.Groups[g]['ytrue']]
        for truth, score in zip(bool_actuals,self.Groups[g]['yhat']):  
            if score > t2:
                if truth:                              # actually positive 
                    tp += 1
                else:                                  # actually negative              
                    fp += 1
                
            elif score < t1:
                if not truth:                          # actually negative 
                    tn += 1                          
                else:                                  # actually positive 
                    fn += 1
                
            elif score>=t1 and score<=t2: 
                mid = np.random.choice(l, p = [1-p2,p2])
                if mid: 
                    if truth: 
                        tp+=1
                    else: fp+=1
                else: 
                    if truth: 
                        fn+=1
                    else: 
                        tn+=1
        return ConfusionMatrix(tp, fp, tn, fn)
    
    
                    
    def conf_(self,t, g, positive_label=1):
        bool_actuals = [act==positive_label for act in self.Groups[g]['ytrue']]
        tp=fp=tn=fn=0
        for truth, score in zip(bool_actuals,self.Groups[g]['yhat']):
            if score < t:
                if truth:                              # actually positive 
                    tp += 1
                else:                                  # actually negative              
                    fp += 1
                
            else:
                if not truth:                          # actually negative 
                    tn += 1                          
                else:                                  # actually positive 
                    fn += 1
    
        return ConfusionMatrix(tp, fp, tn, fn)

    def conf_models(self,t, g, positive_label=1):
        bool_actuals = [act==positive_label for act in self.Groups[g]['ytrue']]
        tp=fp=tn=fn=0
        for truth, score in zip(bool_actuals,self.Groups[g]['yhat']):
            if score > t:
                if truth:                              # actually positive 
                    tp += 1
                else:                                  # actually negative              
                    fp += 1
                
            else:
                if not truth:                          # actually negative 
                    tn += 1                          
                else:                                  # actually positive 
                    fn += 1
    
        return ConfusionMatrix(tp, fp, tn, fn)


    
    def FP_TP_rate(self,conf_mtrx):
        RFPR = conf_mtrx.fp / (conf_mtrx.fp + conf_mtrx.tn) if (conf_mtrx.fp + conf_mtrx.tn)!=0 else 0
        RTPR = conf_mtrx.tp / (conf_mtrx.tp + conf_mtrx.fn) if (conf_mtrx.tp + conf_mtrx.fn)!=0 else 0
        return RFPR, RTPR
    
    
    def ROC_(self,T, models):
        """
        Allthresholds: list of all thresholds of ROC curve. 
        For A=a , allthresholds[i] is the thresholds used to compute (allfpr[i],alltpr[i])
        alltpr and allfpr: for a given ROC curve they define points in the FP, TP plane 
        allauc: AUC of all ROC curves
        """
        ALLfpr, ALLtpr = {}, {}
        
        
        for i,R in enumerate(self.Race): 
            fprl, tprl = [], []
            for t in T: 
                if models:    
                    conf_mtrx = self.conf_models(t, i)
                    
                else: 
                    conf_mtrx = self.conf_(t, i)
                RFPR, RTPR = self.FP_TP_rate(conf_mtrx)
                RFPR = [RFPR]
                RTPR = [RTPR]
                fprl = fprl + RFPR
                tprl = tprl + RTPR 
            ALLfpr[R] = fprl
            ALLtpr[R] = tprl
        
        return ALLfpr, ALLtpr
    
    def acc_with_conf(self, conf_mtrx):
        
        tp = conf_mtrx[0]
        fp = conf_mtrx[1]
        tn = conf_mtrx[2]
        fn = conf_mtrx[3]   

        acc=(tn+tp)/(tn+fp+fn+tp)
        
        return acc


    def acc_(self,T,models,makeplot=True, GetAllOutput=False):
        """
        T: list of thresholds 
        Computes accuracies given lidt of thresholds
        accs: accuracies given list of thresholds
        """

        accs = {}
        if models:
            
            for idx,R in enumerate(self.Race):
                k = []
                for thres in T:
                    conf_mtrx = self.conf_models(thres, idx)
                    tp = conf_mtrx[0]
                    fp = conf_mtrx[1]
                    tn = conf_mtrx[2]
                    fn = conf_mtrx[3]  
                    acc = (tn+tp)/(tn+fp+fn+tp)
                    k.append(acc)
                accs[R] = k
                 
        else:
            for idx,R in enumerate(self.Race):
                k = []   
                for thres in T:
                    conf_mtrx = self.conf_(thres, idx)
                    tp = conf_mtrx[0]
                    fp = conf_mtrx[1]
                    tn = conf_mtrx[2]
                    fn = conf_mtrx[3]   
                    acc = (tn+tp)/(tn+fp+fn+tp)
                    k.append(acc)
                accs[R] = k
                
        return accs

#pandas.core.series.Series

#Name: decile_score.1, Length: 6907, dtype: int64
