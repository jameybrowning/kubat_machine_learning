# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 07:49:48 2017

@author: James Browning
Bayesian Classifier, continuous attributes
following Kubat "An Introduction to Machine Learning"
table 2.6. Example data set is Cleveland Heart data. 13 attributes, 5 classes
(2 classes if consider no heart disease 0 and > 0 heart disease)
"""
#import training set in numpy array
num_row = 297
num_col = 14
import xlrd
import os
import numpy as np
data_full  = np.zeros((num_row,num_col))
os.chdir('C:\\Users\\James Browning\\Google Drive\\Documents\\ML\\Databases') 
book = xlrd.open_workbook("Cleveland_Heart.xlsx")
sheet = book.sheet_by_index(0)
for i in range(0,num_row):
    for j in range(0,num_col):
        data_full[i,j] = sheet.cell_value(i,j)
error_rate = 0
false_pos_rate = 0
false_neg_rate = 0
for n in range(0,num_row):
#test classify first example
    test_ex1 = np.zeros(14)
    test_ex1 = data_full[n,:]
    #need to calculate pos(x)*P(pos) and pneg(x)*P(neg)
    #start by calculating P(pos) and P(neg)
    Ppos = np.count_nonzero(data_full[:,13])
    Pneg = num_row - Ppos
    pposx = pnegx = 1
    #calculate pposatn for each of 13 attributes
    pposat = np.zeros(13)
    pnegat = np.zeros(13)
    for j in range(0,(num_col-1)):
        for i in range(0,num_row):
            if data_full[i,13] > 0:
                pposat[j]=pposat[j]+((1/(Ppos*(2*np.pi)**0.5))*np.exp(-0.5*(test_ex1[j]-data_full[i,j])**2))
            else:
                pnegat[j]=pnegat[j]+((1/(Pneg*(2*np.pi)**0.5))*np.exp(-0.5*(test_ex1[j]-data_full[i,j])**2))
        pposx = pposx*pposat[j]
        pnegx = pnegx*pnegat[j]
    Ppos = Ppos/297
    Pneg = Pneg/297
    Ppostest = pposx*Ppos
    Pnegtest = pnegx*Pneg
    if ((Ppostest > Pnegtest) and (data_full[n,13]==0)):
        error_rate = 1/297+error_rate
        false_pos_rate = 1/297+false_pos_rate
    if ((Ppostest < Pnegtest) and (data_full[n,13]==1)):
        error_rate = 1/297+error_rate
        false_neg_rate = 1/297+false_neg_rate
print("Error Rate = " , 100*error_rate,"%")
print("False Positive Rate = " , 100*false_pos_rate,"%")
print("False Negative Rate = " , 100*false_neg_rate,"%")
# 
        

            
        