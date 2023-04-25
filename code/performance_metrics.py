# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:08:03 2023

@author: egc0021
"""
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import r2_score
import matplotlib as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def confusion_mx(y, y_pred):
    # Get Labels for number of classes 1-numclasses
    labels = pd.Series(y_pred).unique()
    # Get confusion matrix using y=actual and y_pred=predicted
    confusionmx = confusion_matrix(y, y_pred)
    
    # I want to make the cm label in order [-1, 0, 1] so I am tranforming cm
    index_neg1 = np.where(labels == -1)[0][0]
    index_0 = np.where(labels == 0)[0][0]
    index_1 = np.where(labels == 1)[0][0]
    
    ordered_index = [index_neg1, index_0, index_1]
    transformed_cm = np.array([[confusionmx[i][j] for j in ordered_index] for i in ordered_index])
    
    cm_display = ConfusionMatrixDisplay(confusionmx, 
                                        display_labels = list(labels))
    labels = list(labels)
    labels.sort()
    cm_display = ConfusionMatrixDisplay(transformed_cm, 
                                        display_labels = labels)
    cm_display.plot()
    plt.show()
    return cm_display

def r_2(y, y_pred):
    return r2_score(y, y_pred)

def summary(y, y_pred):
    confusion_mx(y, y_pred)
    
    print(classification_report(y, y_pred, labels = y.unique()))
    #print("Accuracy:", sum(y == y_pred)/len(y))
    
    #print("R^2:", r_2(y, y_pred))