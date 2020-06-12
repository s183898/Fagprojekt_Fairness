# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:16:17 2020

@author: mat05
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
date_format = "%Y-%m-%d"

#Comments on loading dataset are obtained from https://arxiv.org/pdf/1906.04711.pdf



#dataset created by ProPublica with purpose of studying two-year general recidivism
#The data set compas-scores-two-years-violent.csv is a subset of violent recidivism
#data_2years = pd.read_csv("./data/compas-scores-two-years.csv")
#data_2years_head = data_2years.columns
#head = data_2years.head
#df = data_2years.drop(columns = ['id', 'name', 'first', 'last', 'sex', 'dob',
 #      'age', 'age_cat', 'race', 'juv_fel_count', 'decile_score',
 #      'juv_misd_count', 'juv_other_count', 'priors_count'])

#print(data_2years.shape)
#data_2yearsv = pd.read_csv("./data/compas-scores-two-years-violent.csv")
#data_2yearsv_head = data_2years.columns
#print(np.sum(data_2years["race"] == "Caucasian"),np.sum(data_2years["race"] == "African-American"))


#full dataset of pretrial defendants that ProPublica obtained from the Broward County Sheriff’s Office
#data_pretrial = pd.read_csv("./data/compas-scores.csv")
#data_pretrial.columns.shape
#data_pretrial_head = data_pretrial.columns.values



#print(np.sum(data_pretrial["race"] == "Caucasian"),np.sum(data_pretrial["race"] == "African-American"))


class dataprocess:
    
    def __init__(self,data_path):
        data = pd.read_csv(data_path)
        self.data = data
        
    def days_len(self,aval,bval,name):
        """ 
        input: 
            aval = start date
            bval = end date
            name = column name of new columns
        
        return: 
            Array with days between aval and bval, or nans if aval/bval = nan
        """
        d = []
        try:
            for i in range(len(self.data[aval])): 
                if type(self.data[aval][i])!=float:
                    a = (datetime.strptime(self.data[aval][i], date_format))
                    b = (datetime.strptime(self.data[bval][i], date_format))
                    d.append((b-a).days)
                else: 
                    d.append(np.nan)
                    
        except ValueError:
            for i in range(len(self.data[aval])): 
                if type(self.data[aval][i])!=float:
                    a = (datetime.strptime(self.data[aval][i].split()[0], date_format))
                    b = (datetime.strptime(self.data[bval][i].split()[0], date_format))
                    d.append((b-a).days)
                else: 
                    d.append(np.nan)
        
        self.data[name] = d
        return d
    
    
    def as_numbers(self,attri):
        """
        input: Categorical attribute consisting of strings 
        function: changes attribute to nummerical data in dataset
        returns: dictionary with strings and corresponding nummerical value
        """

        group = self.data[attri].unique()
        idx = [type(group[i]) for i in range(len(group))]
        if [float in idx]:
            group.remove(group[idx==float])
        dic = {}
        for i,j in enumerate(group):
            dic[j] = i
                
        self.data.replace(dic, inplace=True)
           
        return dic
    
    



##define class variabel
twoyears = dataprocess("./data/compas-scores-two-years.csv")

## Compute legth of stay of compas-score prison time
length  = twoyears.days_len("c_jail_in","c_jail_out",'c_len_of_stay')

##compute nummerical categorical attributes
d_sex = twoyears.as_numbers('sex')
d_agecat = twoyears.as_numbers('age_cat')
d_race = twoyears.as_numbers('race')
d_c_charge_desc = twoyears.as_numbers('c_charge_desc')

 

group = (twoyears.data['r_charge_degree'].unique()).tolist()
idx = [type(group[i]) for i in range(len(group))]
if [float in idx]:
    group.remove(group[idx==float])
dic = {}
for i,j in enumerate(group):
    dic[j] = i
#Data: 
#Compute only the length of the case the compas scores are based on, as
#length of stay of recidivism stay only occurs when recidivism = 1

# Stat of pretrail
#print(np.mean(data_pretrial[data_pretrial["race"] == "Caucasian"]["decile_score.1"]))
#print(np.mean(data_pretrial[data_pretrial["race"] == "African-American"]["decile_score.1"]))