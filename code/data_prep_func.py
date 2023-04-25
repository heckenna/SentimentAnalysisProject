# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:16:23 2023

@author: heckenna
"""

import pandas as pd
import numpy as np

import re

from sklearn.model_selection import train_test_split

# Vectorizer # DONE: Use Tfidf vectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 

# Word forms...
from nltk.stem.wordnet import WordNetLemmatizer




# cd D:\\DDox\\Statistical Learning\\Project\\SentimentAnalysisProject\\code
# cd C:\\Users egc0021\\Documents\\Github\\SentimentAnalysisProject\\code


# Functions for data preperation

### File load or save
def get_data(f_name, f_type = "csv"): # = "Twitter_Data"):
    path = "..\\data\\"
    
    if f_type == "parquet":
        df = pd.read_parquet(path + f_name + ".parquet")
    elif f_type == "csv":  
       df = pd.read_csv(path + f_name + ".csv", sep = ",")
        
    return df

def save_data(df, f_name, f_type = "csv"):
    path = "..\\data\\"
    
    if f_type == "parquet":
        df.to_parquet(path + f_name + ".parquet", index = False)
    elif f_type == "csv":
        df.to_csv(path + f_name, index = False)
    
    return df # Why is save returning the df?

# Need to clean the dataframe and create a column that is ready to be vectorized
# This will probably take a while to run, so make sure to save cleaned data
def clean_df(df, text_col = "clean_text"):
    # First, remove nans
    df =  df.dropna()
    
    #TODO: Need to drop empty strings as well
    
    
    # DONE: Deal with word forms & abbreviations
    
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
                ("modi+j*s?", "modi"),
                ("[a-z]*[0-9]+[a-z]*", " "),
                ("youre", "you are"),
                ("us(ed)?(ing)?e?", "use"),
                ("try?(ing)?(ied)?", "try"),
                ("thats", "that is"),
                ("tak(en)?(ing)?e?", "take"),
                ("successful(ly)?", "successful"),
                ("stop(ped)?(ping)?", "stop"),
                ("start(ed)?(ing)?", "start"),
                ("spread(ing)?", "spread"),
                ("spe(ak(ing)?s?|oke)", "speak"),
                ("isnt?", "is not"),
                ("happen(ed)?s?", "happen"),
                ("follow(ing)?(ed)?", "follow"),
                ("doesnt?", "does not"),
                ("didnt?", "did not"),
                ("demonetis?z?e?(ation)?", "demonetize"),
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
    # return np.vectorize(single_unigram)(col) #DONE: fix setting an array element with sequence
    return [single_unigram(c) for c in col]

# No need. Vectorizer does that for us... breaking into list for preprocessing is good though...
def n_grams(col, n = 1):
    # Note: Start with unigrams
    unigrams = vectorized_single_unigrams(col)
    
    if n == 1:
        return unigrams
    
    return # TO DO: n>1 - No need

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
    # Note: vectorizer needs to be trained on only training data
    # Note: Start with count vectorizer for simplicity sake, but probably want
    #   to see if word2vec or glove does better
    #vectorizer = CountVectorizer(max_features = 5000, min_df = 0.01, max_df = 0.95) 
    
    # Ignore something that appears less than 30 times
    vectorizer = TfidfVectorizer(max_features = 5000, 
                                 stop_words = "english",
                                 min_df = 30/len(col), 
                                 max_df = 0.95,
                                 ngram_range = (1,2))
    vec_col = vectorizer.fit_transform(col)
    
    
    #DONE: Make sure to return vectorizer at some point 
    return vec_col, vectorizer



# DONE. Maybe I should refactor this though.
def create_trainable_feature(df, col_name = "clean_text", vec = None):
    
        
    if vec == None:
        col_fit, vec = train_vectorizer_and_vectorize_column(df[col_name])
    else:
        col_fit = vectorize_col(df[col_name], vec)
        
        
    vect_df = pd.DataFrame(col_fit.todense(), columns=vec.get_feature_names())
    
    df_result = pd.concat([vect_df, df], axis=1)
    
    return df_result, vec


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
    


# DONE: Train-test-validation split
# DONE: Remove punctuation and special characters.  - Started, not finished
# TODO: Build out stop word library
# DONE: Build up regex for abbreviation. Do I want to use Levenshtein distances?- No
# DONE: Save preprocessed data



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
