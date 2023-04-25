# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:18:35 2023

@author: egc0021
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def create_model(model_type = "rf", nest=500, md=20, mss=2, msl=1):
    # Refactoring to make training different models easier.
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=nest, 
                                       max_depth=md,
                                       min_samples_split=mss, 
                                       min_samples_leaf=msl)
    return model 


def create_and_train_model(train_features, train_targs, nest=500, md=20, mss=2, msl=1):
    # DONE: implement this in a data_model_func file
    # Note: Probably want a "train_model" function, 
    #   probably want a "create_model" function as well
    model= create_model("rf", nest, md, mss, msl)
    
    model = model.fit(train_features, train_targs)
    return model

def model_predict(model, train_data):
    pred_train= model.predict(train_data)
    # DONE: implement
    return pred_train

#evaluation metrics of some sort- confusion matrix, EDA
