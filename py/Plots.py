# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:16:17 2020

@author: mat05
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os

#Comments on loading dataset are obtained from https://arxiv.org/pdf/1906.04711.pdf

#dataset created by ProPublica with purpose of studying two-year general recidivism
#The data set compas-scores-two-years-violent.csv is a subset of violent recidivism
data_2years = pd.read_csv("./data/compas-scores-two-years.csv")
data_2years.head()
data_2years_head = data_2years.columns.values

#full dataset of pretrial defendants that ProPublica obtained from the Broward County Sheriff’s Office
data_pretrial = pd.read_csv("./data/compas-scores.csv")
data_pretrial.head()
data_pretrial_head = data_pretrial.columns.values


def histo_func(data, title = " ",  name = "deafult_histogram", directory= 'Images'):
    
    if data == "2 year":
        Caucasian = data_2years[data_2years["race"] == "Caucasian"]["decile_score.1"]
        Black = data_2years[data_2years["race"] == "African-American"]["decile_score.1"]
    elif data == "pretrial":
        Caucasian = data_pretrial[data_pretrial["race"] == "Caucasian"]["decile_score.1"]
        Black = data_pretrial[data_pretrial["race"] == "African-American"]["decile_score.1"]
    else:
        print("Could not save histogram")
        return

    plt.hist([Caucasian,Black], label = ["Caucasian", "African-American"], bins = [i - 0.5 for i in range(1,12)], color= ["C0", "C1"])
    plt.xlabel("Decile score", fontsize=10)  
    plt.ylabel("Number of defendants", fontsize=10)
    plt.xticks(range(11))
    plt.legend(["Caucasian", "African-American"])
    plt.title(title, fontdict= {'fontsize': 14})

    path = directory + '\\' + name + '.png'

    plt.savefig(path)

    print('Histogram saved')

    return

histo_func(data = '2 year', title = '2 year decile score', name = '2year_hist')

histo_func(data = 'pretrial', title = 'pretrial decile score', name = 'pretrial_hist')


