# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:32:30 2020

@author: Værksted Matilde
"""
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_conf(conf_mtrx, title):
    """
    input: confusion matrix of type conf(tp, fp, tn, fn)
    """
    conf = np.empty([2,2])
    conf[0,0] = conf_mtrx[0]
    
    conf[0,1] = conf_mtrx[1]
    conf[1,0] = conf_mtrx[3]
    conf[1,1] = conf_mtrx[2]
    
    df = pd.DataFrame(conf, index = ["Predictive (1)", "Predictive (0)"], columns = ["Actual (1)","Actual (0)"])

    ax = plt.axes()
    sns.heatmap(df/np.sum(df), annot=True, fmt='.2%', cmap='Blues', ax = ax)
    
    ax.set_title(title)

    plt.show()
    return conf 
    
    """
        input to functions below: 
        confusion matrix of type conf(tp, fp, tn, fn)
    """
def ppv(conf_mtrx):
  
    tp = conf_mtrx[0]
    fp = conf_mtrx[1]
    
    ppv = tp/(tp+fp)    
    return ppv

def tdr(conf_mtrx):
  
    tp = conf_mtrx[0]
    fn = conf_mtrx[3]
    
    tdr = tp/(tp+fn)    
    return tdr
    
def fOr(conf_mtrx):

    tn = conf_mtrx[2]
    fn = conf_mtrx[3]
    
    fOr = fn/(tn+fn)    
    return fOr
       
def fnr(conf_mtrx):
  
    tp = conf_mtrx[0]
    fn = conf_mtrx[3]
    
    fnr = fn/(tp+fn)    
    return fnr    

def fdr(conf_mtrx):
  
    tp = conf_mtrx[0]
    fp = conf_mtrx[1]
    
    fdr = fp/(tp+fp)    
    return fdr 
    
def fpr(conf_mtrx):

    fp = conf_mtrx[1]
    tn = conf_mtrx[2]
    
    fpr = fp/(fp+tn)   
    return fpr
    
def npv(conf_mtrx):
  
    fp = conf_mtrx[1]
    tn = conf_mtrx[2]
    
    npv = tn/(tn+fp)   
    return npv
    
def tnr(conf_mtrx):

    fp = conf_mtrx[1]
    tn = conf_mtrx[2]
    
    tnr = tn/(tn+fp)   
    return tnr    
    
    

    