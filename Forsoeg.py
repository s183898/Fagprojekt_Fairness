# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:16:17 2020

@author: mat05
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#Comments on loading dataset are obtained from https://arxiv.org/pdf/1906.04711.pdf



#dataset created by ProPublica with purpose of studying two-year general recidivism
#The data set compas-scores-two-years-violent.csv is a subset of violent recidivism
data_2years = pd.read_csv("./compas-scores-two-years.csv")
data_2years.head()
data_2years_head = data_2years.columns.values

#print(data_2years.shape)

#print(np.sum(data_2years["race"] == "Caucasian"),np.sum(data_2years["race"] == "African-American"))


#full dataset of pretrial defendants that ProPublica obtained from the Broward County Sheriff’s Office
data_pretrial = pd.read_csv("./compas-scores.csv")
data_pretrial.head()
data_pretrial_head = data_pretrial.columns.values


# Hist of pretrail

#plt.subplot(2,1,1)
#plt.hist(data_pretrial[data_pretrial["race"] == "Caucasian"]["decile_score.1"])

#plt.subplot(2,1,2)
#plt.hist(data_pretrial[data_pretrial["race"] == "African-American"]["decile_score.1"])

#plt.show()

# Stat of pretrail

print(np.mean(data_pretrial[data_pretrial["race"] == "Caucasian"]["decile_score.1"]))
print(np.mean(data_pretrial[data_pretrial["race"] == "African-American"]["decile_score.1"]))

# Plot of 2year

#Caucasian = data_pretrial[data_pretrial["race"] == "Caucasian"]["decile_score.1"]
#Black = data_pretrial[data_pretrial["race"] == "African-American"]["decile_score.1"]

Caucasian = data_2years[data_2years["race"] == "Caucasian"]["decile_score.1"]
Black = data_2years[data_2years["race"] == "African-American"]["decile_score.1"]


plt.hist([Caucasian,Black], label = ["Caucasian", "African-American"], bins = [i - 0.5 for i in range(1,12)])

plt.xlabel("Decile score", fontsize=10)  
plt.ylabel("Number of convicted", fontsize=10)

plt.xticks(range(11))

plt.legend()
plt.title("2 year decile scores", fontdict= {'fontsize': 14})
plt.show()


#print(np.sum(data_pretrial["race"] == "Caucasian"),np.sum(data_pretrial["race"] == "African-American"))

