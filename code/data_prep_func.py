# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:16:23 2023

@author: heckenna
"""

import pandas as pd
import numpy as np

import re

from sklearn.model_selection import train_test_split

# Vectorizer # TODO: Use Tfidf vectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 


# cd D:\\DDox\\Statistical Learning\\Project\\SentimentAnalysisProject\\code

# Functions for data preperation
def get_data(f_name): # = "Twitter_Data"):
    path = "..\\data\\"
    
    #TODO implement this in a data_prep_func fie
    df = pd.read_csv(path + f_name + ".csv", sep = ",")    
    return df

# Need to clean the dataframe and create a column that is ready to be vectorized
# This will probably take a while to run, so make sure to save cleaned data
def clean_df(df, text_col = "clean_text"):
    # First, remove nans
    df =  df.dropna()
    
    #TODO: Need to drop empty strings as well
    
    df["happy_col"] = clean_feature_col(df[text_col])
    
    return df


def clean_feature_col(dirty_col):        
    # Probably need to remove stop words and have abbreviation library
    dirty_col = vec_space_replacements(dirty_col)
    dirty_col = vec_no_space_replacements(dirty_col)
    
    # Remove stop words next
    dirty_col = vec_remove_stops(dirty_col)
    
    
    # Probably design a list replacement thing in order to deal with abbreviations...
    
    
    return dirty_col

def single_unigram(row):
    row_list = row.split(" ")
    return row_list

def vectorized_single_unigrams(col):
    # return np.vectorize(single_unigram)(col) #TODO: fix setting an array element with sequence
    return [single_unigram(c) for c in col]


### Theis set of functions replaces certain elements with spaces or deletes them
def do_replacements(row_text, replacement, pat):
    return re.sub(pattern = pat, repl = replacement, string = row_text)
   
def vec_space_replacements(col):
    reps = []
    
    pat = re.compile(" |\\n|\n|\.|,") 
    
    # Currying for the curious
    vec_func = np.vectorize(lambda row_text: do_replacements(row_text, " ", pat))
    
    return vec_func(col)
    
   
def vec_no_space_replacements(col):
    reps = []
    # Need to deal with [0-9]+
    
    pat = re.compile("\"|\'|\”|\“|’|‘") 
    
    # Currying for the curious
    vec_func = np.vectorize(lambda row_text: do_replacements(row_text, "", pat))
    
    return vec_func(col)

def vec_remove_stops(col):
    
    stop_words_list = ["(?=[\W\Z])" + "the", 
                       "a", 
                       "an" + "(?<=[\W\A])"]
    
    # Take list and turn into string which can be used for regex pattern matching
    #   The "|" means "or"
    #   The stuff in parentheses says "I want to match the word 'an' not any 
    #       time I see 'an' together inside a word"
    #       (for example, dont remove 'an' from 'any')
    stop_words_str = "(?=[\W\Z])|(?<=[\W\A])".join(stop_words_list)
    
    pat = re.compile(stop_words_str) 
    
    # Currying for the curious
    vec_func = np.vectorize(lambda row_text: do_replacements(row_text, "", pat))
    
    return vec_func(col)


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
    vectorizer = CountVectorizer() #TfidfVectorizer()
    vec_col = vectorizer.fit_transform(col)
    
    #TODO: Make sure to return vectorizer at some point 
    return vec_col, vectorizer

def save_data(df, f_name):
    # TODO: implement this in a data_prep_func file
    # This can be used to save preprocessed data
    path = "..\\data\\"
    
    return


def create_trainable_feature(df, col = "clean_text"):
    df["feature"] = create_vectorized_column(df["clean_text"])
    
    return df


def train_test_val_split(df, random_state= 9632, targ = "category"):
    # Using random state in order to be deterministically random
    train, test = train_test_split(df, 
                                   test_size=0.4, 
                                   random_state= 9632, 
                                   stratify = df[targ])
    test, val = train_test_split(test, 
                                 test_size=0.5, 
                                 random_state= 9632, 
                                 stratify = test[targ])
    return train, test, val
    


# TODO: Train-test-validation split
# TODO: Remove punctuation and special characters.  - Started, not finished
# TODO: Build out stop word library
# TODO: Build up regex for abbreviation. Do I want to use Levenshtein distances?
# TODO: Save preprocessed data



#Testing my functions
df = get_data("Twitter_Data")

df = df.dropna()

train_df, test_df, val_df = train_test_val_split(df)

#head_df = df.head(5000)


#head_df["feature"] = create_vectorized_column(head_df["clean_text"])

#col = vec_space_replacements(head_df["clean_text"])
#col1 = vec_no_space_replacements(head_df["clean_text"])
#col = clean_feature_col(head_df["clean_text"])


#cleaner_df = clean_df(head_df)
