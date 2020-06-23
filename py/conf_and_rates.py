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
    sns.heatmap(df, annot=True, fmt='.4', cmap='Blues',annot_kws={"size": 11}, ax = ax)
    
    ax.set_title(title)

    plt.show()
    return conf 
    
    """
        input to functions below: 
        confusion matrix of type conf(tp, fp, tn, fn)
    """
def rates(conf_mtrx):
  
    tp = conf_mtrx[0]
    fp = conf_mtrx[1]
    tn = conf_mtrx[2]
    fn = conf_mtrx[3]
    
    PPV = tp/(tp+fp) 

    FOR = fn/(tn+fn) #
    FNR = fn/(tp+fn) 
    FDR = fp/(tp+fp) #
    NPV = tn/(tn+fn) #
    TNR = tn/(tn+fp)
    return [PPV, FOR, FNR, FDR ,NPV, TNR]

def baserate(conf_mtrx): 
    tp = conf_mtrx[0]
    fp = conf_mtrx[1]
    tn = conf_mtrx[2]
    fn = conf_mtrx[3]
    
    SBR = 1
    FBR = 1
    
    return [SBR, FBR]