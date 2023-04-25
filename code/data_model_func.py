# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:18:35 2023

@author: egc0021
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def create_model(model_type = "rf", nest=200, md=20, mss=20, msl=10):
    # Refactoring to make training different models easier.
    # Using random state to seed models so reproduceable.
    if model_type == "rf":
        model = RandomForestClassifier(random_state = 7893,
                                       n_estimators=nest, 
                                       max_depth=md,
                                       min_samples_split=mss, 
                                       min_samples_leaf=msl,
                                       n_jobs = -1)
    elif model_type == "gb":
        model = GradientBoostingClassifier(random_state = 7893,
                                           n_estimators=nest, 
                                           max_depth=md,
                                           min_samples_split=mss, 
                                           min_samples_leaf=msl)
    elif model_type == "nn":
        model= MLPClassifier(solver='lbfgs', 
                             alpha=1e-5,
                             hidden_layer_sizes=(10, 10), 
                             random_state=1)
    
    return model 


def create_and_train_model(train_features, train_targs, model_type = "rf", nest=500, md=20, mss=20, msl=10):
    # DONE: implement this in a data_model_func file
    # Note: Probably want a "train_model" function, 
    #   probably want a "create_model" function as well
    model= create_model(model_type, nest, md, mss, msl)
    
    model = model.fit(train_features, train_targs)
    return model

def model_predict(model, train_data):
    pred_train= model.predict(train_data)
    # DONE: implement
    return pred_train

#evaluation metrics of some sort- confusion matrix, EDA
