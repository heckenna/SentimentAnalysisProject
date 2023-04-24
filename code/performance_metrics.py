# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:08:03 2023

@author: egc0021
"""
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib as plt
import pandas as pd

def confusion_mx(x, y, y_pred, numcl):
    # Get Labels for number of classes 1-numclasses
    labels = list(range(1,numcl+1))
    # Get confusion matrix using y=actual and y_pred=predicted
    confusion_mx = confusion_matrix(y, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_mx = confusion_mx, 
                                                display_labels = list(labels))
    cm_display.plot()
    plt.show()
    return cm_display