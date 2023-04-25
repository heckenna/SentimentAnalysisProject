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

train_targ = train_df["CATEGORY"]
test_targ = test_df["CATEGORY"]
val_targ = val_df["CATEGORY"]

train_df.drop(columns = ["clean_text", "CATEGORY"], inplace = True)
test_df.drop(columns = ["clean_text", "CATEGORY"], inplace = True)
val_df.drop(columns = ["clean_text", "CATEGORY"], inplace = True)
#'''


# Train model
m_t = "gb"
print("Training model")
print("Type:", m_t)
start = time.time()
model = dmf.create_and_train_model(train_df, train_targ, model_type = m_t)
end  = time.time()
print("Trained in", end - start, "seconds.")

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
train_preds = dmf.model_predict(model, train_df)
test_preds = dmf.model_predict(model, test_df)
val_preds = dmf.model_predict(model, val_df)
 
# Evaluation metrics
#goodness_of_fit(train_preds, train_targs)

#foo
#confusion matrix
train_conf_mx = pm.summary(train_targ, train_preds)
test_conf_mx = pm.summary(test_targ, test_preds)
#print(conf_mx)





