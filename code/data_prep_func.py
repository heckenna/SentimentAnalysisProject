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

# Word forms...
from nltk.stem.wordnet import WordNetLemmatizer




# cd D:\\DDox\\Statistical Learning\\Project\\SentimentAnalysisProject\\code

# Functions for data preperation

### File load or save
def get_data(f_name): # = "Twitter_Data"):
    path = "..\\data\\"
    
    df = pd.read_csv(path + f_name + ".csv", sep = ",")    
    return df

def save_data(df, f_name):
    path = "..\\data\\"
    
    df.to_csv(path + f_name, index = False)
    return df

# Need to clean the dataframe and create a column that is ready to be vectorized
# This will probably take a while to run, so make sure to save cleaned data
def clean_df(df, text_col = "clean_text"):
    # First, remove nans
    df =  df.dropna()
    
    #TODO: Need to drop empty strings as well
    
    
    # TODO: Deal with word forms & abbreviations
    
    df["vectorizable_text"] = clean_feature_col(df[text_col])
    
    return df


def clean_feature_col(dirty_col):        
    # Probably need to remove stop words and have abbreviation library
    dirty_col = vec_space_replacements(dirty_col)
    dirty_col = vec_no_space_replacements(dirty_col)
    
    # Remove stop words next
    dirty_col = vec_remove_stops(dirty_col)
    
    
    #### Probably design a list replacement thing in order to deal with abbreviations...
    tokenized_col = vectorized_single_unigrams(dirty_col)
    
    
    # Deal with different word forms
    tokenized_col = lemmatize_col(tokenized_col)
    
    # Make substitutions myself
    tokenized_col = make_word_subs_col(tokenized_col)
    
    clean_col = vec_list_to_string(tokenized_col)
    return clean_col #" ".join(tokenized_col)


#### List to string
def vec_list_to_string(col):
    return np.vectorize(lambda row: " ".join(row))(col)



### Lemmatizer
def lemmatize_row(lemmatizer, row):
    return [lemmatizer.lemmatize(word) for word in row]
    
def lemmatize_col(col):
    lemmatizer = WordNetLemmatizer()
    helper =  lambda row: lemmatize_row(lemmatizer, row)
    return [helper(row) for row in col]


### Similar to lemmatization, replaces abbreviations/etc with actual words

def make_word_sub(word):
    # Object of things that need swapped. Each tuple (a, b) inside the list
    #   means to replace "a" with "b"
    # TODO: Maybe use a hashmap here?
    matchers = [("tho", "though"),
                ("govt?", "government"),
                ("pl[sz]", "please"),
                ("(ha)+", "ha"),
                ("shou+l?d+", "should"),
                ("yrs?", "year"),
                ("wtf", "fuck"),
                ("m", "meter"),
                ]
    
    
    for match in matchers:
        if re.fullmatch(match[0], word):
            return match[1]
        
    return word
    

def make_word_subs_row(row):
    return [make_word_sub(word) for word in row]

def make_word_subs_col(col):
    return [make_word_subs_row(row) for row in col]




### This tokenizes the data
def single_unigram(row):
    row_list = row.split(" ")
    return row_list

def vectorized_single_unigrams(col):
    # return np.vectorize(single_unigram)(col) #TODO: fix setting an array element with sequence
    return [single_unigram(c) for c in col]

# No need. Vectorizer does that for us... breaking into list for preprocessing is good though...
def n_grams(col, n = 1):
    # TODO implement this in a data_prep_func file
    # Note: Start with unigrams
    unigrams = vectorized_single_unigrams(col)
    
    if n == 1:
        return unigrams
    
    return # TODO: n>1

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
    
    stop_words_list = ["(?=\W)" + "the", 
                       "a",
                       "an" + "(?<=\W)"]
    
    # Take list and turn into string which can be used for regex pattern matching
    #   The "|" means "or"
    #   The stuff in parentheses says "I want to match the word 'an' not any 
    #       time I see 'an' together inside a word"
    #       (for example, dont remove 'an' from 'any')
    stop_words_str = "(?=\W)|(?<=\W)".join(stop_words_list)
    
    pat = re.compile(stop_words_str) 
    
    # Currying for the curious
    vec_func = np.vectorize(lambda row_text: do_replacements(row_text, "", pat))
    
    return vec_func(col)



def vectorize_col(col, vectorizer):
    return vectorizer.transform(col)

def train_vectorizer_and_vectorize_column(col):
    # TODO implement this in a data_prep_func file
    # Note: vectorizer needs to be trained on only training data
    # Note: Start with count vectorizer for simplicity sake, but probably want
    #   to see if word2vec or glove does better
    vectorizer = CountVectorizer() 
    #vectorizer = TfidfVectorizer()
    vec_col = vectorizer.fit_transform(col)
    
    #TODO: Make sure to return vectorizer at some point 
    return vec_col, vectorizer




def create_trainable_feature(df, col = "clean_text"):
    df["feature"], vec = create_vectorized_column(df["clean_text"])
    
    return df


# Splits data into train, test and validation. 
# TODO: Let user determine split
def train_test_val_split(df, random_state= 9632, targ = "category"):
    # Using random state in order to be deterministically random
    df = df.dropna()
    
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
#df = get_data("Twitter_Data")

#df = df.dropna()

#train_df, test_df, val_df = train_test_val_split(df)

#head_df = df.head(5000)


#head_df["feature"] = create_vectorized_column(head_df["clean_text"])

#col = vec_space_replacements(head_df["clean_text"])
#col1 = vec_no_space_replacements(head_df["clean_text"])
#col = clean_feature_col(head_df["clean_text"])


#cleaner_df = clean_df(head_df)


#save_data(cleaner_df, "clean_test.csv")
