# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:42:37 2023

@author: heckenna
"""

import data_prep_func as dpf



### Vectorizes Data


# Get data
train_df = dpf.get_data("initial_train_unvectorized")
test_df = dpf.get_data("initial_test_unvectorized")
val_df = dpf.get_data("initial_val_unvectorized")


# Vectorize data
train_df["feature"], vec = dpf.train_vectorizer_and_vectorize_column(train_df["vectorizable_text"])
test_df["feature"] = dpf.vectorize_col(test_df["vectorizable_text"], vec)
val_df["feature"] = dpf.vectorize_col(val_df["vectorizable_text"], vec)


# Save data
dpf.save_data(train_df, "initial_train_vectorized.csv")
dpf.save_data(test_df, "initial_test_vectorized.csv")
dpf.save_data(val_df, "initial_val_vectorized.csv")

 