# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:16:23 2023

@author: heckenna
"""

import pandas as pd
import numpy as np

import re

from sklearn.feature_extraction.text import CountVectorizer
# TODO: Used Tfidf vectorizer

# cd D:\\DDox\\Statistical Learning\\Project\\SentimentAnalysisProject\\code

# Functions for data preperation
def get_data(f_name): # = "Twitter_Data"):
    path = "..\\data\\"
    
    #TODO implement this in a data_prep_func fie
    df = pd.read_csv(path + f_name + ".csv", sep = ",")    
    return df


def clean_data(data):
    # TODO implement this in a data_prep_func file
    # Probably need to remove stop words and have abbreviation library
    
    
    return

def single_unigram(row):
    row_list = row.split(" ")
    return row_list

def vectorized_single_unigrams(col):
    #return np.vectorize(single_unigram)(col) #TODO: fix setting an array element with sequence
    return [single_unigram(c) for c in col]

# No need. Vectorizer does that for us... breaking into list for preprocessing is good though...
def n_grams(col, n = 1):
    # TODO implement this in a data_prep_func file
    # Note: Start with unigrams
    unigrams = vectorized_single_unigrams(col)
    
    if n == 1:
        return unigrams
    
    return # TODO: n>1

def create_vectorized_column(col):
    # TODO implement this in a data_prep_func file
    # Note: vectorizer needs to be trained on only training data
    # Note: Start with count vectorizer for simplicity sake, but probably want
    #   to see if word2vec or glove does better
    vectorizer = CountVectorizer()
    vec_col = vectorizer.fit_transform(col)
    
    #TODO: Make sure to return vectorizer at some point 
    return vec_col

def save_data(df, filename):
    # TODO: implement this in a data_prep_func file
    # This can be used to save preprocessed data
    return


def create_trainable_feature(df, col = "clean_text"):
    df["feature"] = create_vectorized_column(df["clean_text"])
    
    return df



#Testing my functions
#df = get_data("Twitter_Data")

#head_df = df.head(100)


#head_df["feature"] = create_vectorized_column(head_df["clean_text"])





