# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:42:37 2023

@author: heckenna
"""

import data_prep_func as dpf



### Vectorizes Data


# Get data
f_type = "parquet"
train_df = dpf.get_data("initial_train_unvectorized", f_type)
test_df = dpf.get_data("initial_test_unvectorized", f_type)
val_df = dpf.get_data("initial_val_unvectorized", f_type)


# Vectorize data
#train_df["feature"], vec = dpf.train_vectorizer_and_vectorize_column(train_df["vectorizable_text"])
#test_df["feature"] = dpf.vectorize_col(test_df["vectorizable_text"], vec)
#val_df["feature"] = dpf.vectorize_col(val_df["vectorizable_text"], vec)

train_df = train_df.rename(columns={"category": "CATEGORY"})
test_df = test_df.rename(columns={"category": "CATEGORY"})
val_df = val_df.rename(columns={"category": "CATEGORY"})

train_df, vec = dpf.create_trainable_feature(train_df, 
                                             col_name  = "vectorizable_text")
test_df, vec = dpf.create_trainable_feature(test_df, 
                                            col_name  = "vectorizable_text",
                                            vec = vec)
val_df, vec = dpf.create_trainable_feature(val_df, 
                                           col_name  = "vectorizable_text", 
                                           vec = vec)

#'''

# Save data
print("Saving data")
print("Saving train")
dpf.save_data(train_df, 
              "initial_train_vectorized",
              f_type)
print("Saving test")
dpf.save_data(test_df, 
              "initial_test_vectorized",
              f_type)
print("Saving val")
dpf.save_data(val_df, 
              "initial_val_vectorized",
              f_type)

 