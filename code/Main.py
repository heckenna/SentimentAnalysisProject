# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 09:19:55 2023

@author: heckenna
"""

# The main code to run will wrap the code in this file in a "main" function 
import data_prep_func as dpf
import data_model_func as dmf
import numpy as np
import performance_metrics as pm
import time



###############################################
#
# This is a code skeleton so far
#
###############################################


# TODO: EDA in some form needs done. 
#   Take a look at data to see what is going on
#   Probably some of the summary code could be written in order to do EDA


# Get data
print("Getting Data")
f_type = "parquet"
train_df = dpf.get_data("initial_train_vectorized", f_type)
test_df = dpf.get_data("initial_test_vectorized", f_type)
val_df = dpf.get_data("initial_val_vectorized", f_type)

train_feat = train_df.drop(columns = ["clean_text", "CATEGORY"])
test_feat = test_df.drop(columns = ["clean_text", "CATEGORY"])
val_feat = val_df.drop(columns = ["clean_text", "CATEGORY"])

train_targ = train_df["CATEGORY"]
test_targ = test_df["CATEGORY"]
val_targ = val_df["CATEGORY"]

'''
# TODO: remove this. Just making sure the rest runs.
train_data = data #.head(80000)

train_data = train_data.rename(columns={"category": "CATEGORY"})
#features, vec = dpf.train_vectorizer_and_vectorize_column(train_data["vectorizable_text"])
start = time.time()
train_data, vec = dpf.create_trainable_feature(train_data, col_name  = "vectorizable_text")
end = time.time()
tot = end - start
print("Vectorization in", tot, "seconds.")

print("Saving data")
dpf.save_data(train_data, "size_checker")
#'''

# Train model
print("Training model")
model = dmf.create_and_train_model(train_feat, train_targ)


# TBH, probably want to preprocess in a different file and save

# Clean data
#data = clean_data(data)

# Split into n-grams (start with unigrams at first)
#data = n_grams(data)

# Vectorize n_grams (Probably count at first)
#data = create_vectorized_column(data)


# Run model on data
#model = data_model_func.create_and_train_model(features.toarray(), train_data["category"])

# Predict on train-test (val)
print("Predicting with model")
train_preds = dmf.model_predict(model, train_feat)
test_preds = dmf.model_predict(model, test_feat)
val_preds = dmf.model_predict(model, val_feat)
 
# Evaluation metrics
#goodness_of_fit(train_preds, train_targs)

#foo
#confusion matrix
conf_mx=pm.confusion_mx(train_targ, train_preds)
print(conf_mx)





