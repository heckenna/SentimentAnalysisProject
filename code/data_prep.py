# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:10:04 2023

@author: heckenna
"""
import data_prep_func as dpf

##### Does preprocessing and creates train-test-validatation files.
# Such data has "vectorizable_text" column which just needs vectorized in 
#   order to be used

# Get the data
df = dpf.get_data("Twitter_Data", "csv")

# Clean data. Remove stops, fix abbreviations, lemmatize
print("Cleaning")
start = time.time()
df = dpf.clean_df(df, text_col = "clean_text")
end = time.time()
tot = end - start
print("Took", tot, "seconds.")


# Split the data
train_df, test_df, val_df = dpf.train_test_val_split(df)


# Save the data
print("Saving")
f_type = "parquet"
dpf.save_data(train_df, "initial_train_unvectorized", f_type)
dpf.save_data(test_df, "initial_test_unvectorized", f_type)
dpf.save_data(val_df, "initial_val_unvectorized", f_type)

