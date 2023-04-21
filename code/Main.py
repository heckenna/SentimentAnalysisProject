# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 09:19:55 2023

@author: heckenna
"""

# The main code to run will wrap the code in this file in a "main" function 
import data_prep_func

### These need to be in data_prep_func
def get_data(filename):
    path = ""
    #TODO implement this in a data_prep_func fie
    return

def clean_data(data):
    # TODO implement this in a data_prep_func file
    # Probably need to remove stop words and have abbreviation library
    return

def n_grams(data, n = 1):
    # TODO implement this in a data_prep_func file
    # Note: Start with unigrams
    return

def create_vectorized_column(data):
    # TODO implement this in a data_prep_func file
    # Note: vectorizer needs to be trained on only training data
    # Note: Start with count vectorizer for simplicity sake, but probably want
    #   to see if word2vec or glove does better
    return

def save_data(df, filename):
    # TODO: implement this in a data_prep_func file
    # This can be used to save preprocessed data
    return


### These need to be in data_model_func
def create_and_train_model(train_features, train_targs):
    # TODO: implement this in a data_model_func file
    # Note: Probably want a "train_model" function, 
    #   probably want a "create_model" function as well
    return

def model_predict(model, train_data):
    # TODO: implement
    return

### These need implemented in a summary file
def goodness_of_fit():
    #TODO: implement
    return


###############################################
#
# This is a code skeleton so far
#
###############################################


# TODO: EDA in some form needs done. 
#   Take a look at data to see what is going on
#   Probably some of the summary code could be written in order to do EDA


# Get data
data = data_prep_func.get_data("Twitter_Data")

# TODO: remove this. Just making sure the rest runs.
data = data.head(100)


data = data_prep_func.create_trainable_feature(data)
# TBH, probably want to preprocess in a different file and save

# Clean data
#data = clean_data(data)

# Split into n-grams (start with unigrams at first)
#data = n_grams(data)

# Vectorize n_grams (Probably count at first)
#data = create_vectorized_column(data)

# Run model on data
model = create_and_train_model(train_data["vac_column"], train_data["targ"])

# Predict on train-test (val)
train_preds = predict(model, train_data)
test_preds = predict(model, test_data)
val_preds = predict(model, val_data)

# Evaluation metrics
goodness_of_fit(train_preds, train_targs)

#foo





