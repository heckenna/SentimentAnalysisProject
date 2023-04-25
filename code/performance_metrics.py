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
import matplotlib.pyplot as plt


def confusion_mx(y, y_pred):
    # Get Labels for number of classes 1-numclasses
    labels = y.unique()
    # Get confusion matrix using y=actual and y_pred=predicted
    confusionmx = confusion_matrix(y, y_pred)
    cm_display = ConfusionMatrixDisplay(confusionmx, 
                                        display_labels = list(labels))
    cm_display.plot()
    plt.show()
    return cm_display