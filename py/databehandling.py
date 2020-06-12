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
data_2years = pd.read_csv("./data/compas-scores-two-years.csv")
data_2years_head = data_2years.columns
head = data_2years.head
df = data_2years.drop(columns = ['id', 'name', 'first', 'last', 'sex', 'dob',
       'age', 'age_cat', 'race', 'juv_fel_count', 'decile_score',
       'juv_misd_count', 'juv_other_count', 'priors_count'])

#print(data_2years.shape)
data_2yearsv = pd.read_csv("./data/compas-scores-two-years-violent.csv")
data_2yearsv_head = data_2years.columns
#print(np.sum(data_2years["race"] == "Caucasian"),np.sum(data_2years["race"] == "African-American"))


#full dataset of pretrial defendants that ProPublica obtained from the Broward County Sheriff’s Office
data_pretrial = pd.read_csv("./data/compas-scores.csv")
data_pretrial.columns.shape
data_pretrial_head = data_pretrial.columns.values


# Hist of pretrail
meanc =(np.mean(data_pretrial[data_pretrial["race"] == "Caucasian"]["decile_score.1"]))
meana = (np.mean(data_pretrial[data_pretrial["race"] == "African-American"]["decile_score.1"]))

d1 = data_pretrial[data_pretrial["race"] == "Caucasian"]["decile_score.1"]
d2 = data_pretrial[data_pretrial["race"] == "African-American"]["decile_score.1"]
plt.hist([d1,d2], histtype = 'bar')
plt.title("Distribution of Risk of Recidivism score, pretrial data set")
plt.xlabel("Risk of Recidivism: Score")
plt.ylabel("Number of defendants")
plt.legend(('Caucasian defendents ($\mu$ = %.2f)' % meanc, 'African-American defendents ($\mu$ = %.2f)' %meana),
           shadow=True, loc=(0.2, 0.8), fontsize=9)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9,10])
plt.show()

# Plot of 2year
meanc1 = (np.mean(data_2years[data_2years["race"] == "Caucasian"]["decile_score.1"]))
meana1 = (np.mean(data_2years[data_2years["race"] == "African-American"]["decile_score.1"]))

d1 = data_2years[data_2years["race"] == "Caucasian"]["decile_score"]
d2 = data_2years[data_2years["race"] == "African-American"]["decile_score"]
plt.hist([d1,d2], histtype = 'bar')
plt.title("Distribution of Risk of Recidivism score, two years recidivism data set")
plt.xlabel("Risk of Recidivism: Score")
plt.ylabel("Number of defendants")
plt.legend(('Caucasian defendents ($\mu$ = %.2f)' % meanc1, 'African-American defendents ($\mu$ = %.2f)' %meana1),
           shadow=True, loc=(0.2, 0.8), fontsize=9)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9,10])
#plt.show()
#decile_score.1
#plt.subplot(2,1,1)
#plt.hist(data_pretrial[data_pretrial["race"] == "Caucasian"]["decile_score.1"])

#plt.subplot(2,1,2)
#plt.hist(data_pretrial[data_pretrial["race"] == "African-American"]["decile_score.1"])

#plt.show()

# Stat of pretrail
#print(np.mean(data_pretrial[data_pretrial["race"] == "Caucasian"]["decile_score.1"]))
#print(np.mean(data_pretrial[data_pretrial["race"] == "African-American"]["decile_score.1"]))

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
#plt.show()


#print(np.sum(data_pretrial["race"] == "Caucasian"),np.sum(data_pretrial["race"] == "African-American"))
from datetime import datetime
date_format = "%Y-%m-%d"

#diff of r_jail_out and r_jail_in
diff_r = []
for i in range(len(data_pretrial["r_jail_out"])): 
    if type(data_pretrial["r_jail_out"][i])!=float:
        a = (datetime.strptime(data_pretrial["r_jail_out"][i], date_format))
        b = (datetime.strptime(data_pretrial["r_jail_in"][i], date_format))
        diff_r.append((a-b).days)

#diff of c_jail_out and c_jail_in
diff_c = []
for i in range(len(data_pretrial["c_jail_out"])): 
    if type(data_pretrial["c_jail_out"][i])!=float:
        a = (datetime.strptime(data_pretrial["c_jail_out"][i].split()[0], date_format))
        b = (datetime.strptime(data_pretrial["c_jail_in"][i].split()[0], date_format))
        diff_c.append((a-b).days)
        


#diff of c_jail_out and c_jail_in
diff_l = []
for i in range(len(data_pretrial["compas_screening_date"])): 
    if type(data_pretrial["compas_screening_date"][i])!=float and type(data_pretrial["c_jail_in"][i])!=float:
        a = (datetime.strptime(data_pretrial["compas_screening_date"][i], date_format))
        b = (datetime.strptime(data_pretrial["c_jail_in"][i].split()[0], date_format))
        diff_l.append((b-a).days)
        
clean = data_2years["days_b_screening_arrest"].to_list()
clean = [x for x in clean if type(x)!=float]