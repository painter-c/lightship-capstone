from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import FitFailedWarning

import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
import string

class BlacklistFilter(BaseEstimator, TransformerMixin):
    
    # The blacklist must be a list of strings
    def __init__(self, blacklist, column):
        self.blacklist = blacklist
        self.column = column
        
    # X must be a dataframe with the 'assignee_id' column
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise FitFailedWarning('X must be an instance of pandas.Dataframe.')
        elif self.column not in X.columns:
            raise FitFailedWarning(f'X does not have the {self.column} column.')
        else:
            return self
        
    def transform(self, X, y=None):
        X_new = X
        for account_id in self.blacklist:
            X_new = X[X[self.column].ne(account_id)]
        return X_new
    
    
class NullEntryFilter(BaseEstimator, TransformerMixin):
    
    def __init__(self, column):
        self.column = column
        
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise FitFailedWarning('X must be an instance of pandas.Dataframe.')
        elif self.column not in X.columns:
            raise FitFailedWarning(f'X does not have the {self.column} column.')
        else:
            return self
        
    def transform(self, X, y=None):
        X_new = X[X[self.column].notnull()]
        return X_new
    
    
class HashDecoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, keyword_table, columns):
        self.keyword_table = keyword_table
        self.columns = columns
        
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise FitFailedWarning('X must be an instance of pandas.Dataframe.')
        return self
        
    def __decode_string(self, string):
        tokens = string.split(' ')
        decoded = [self.keyword_table[token] for token in tokens if token in self.keyword_table]
        return ' '.join(decoded)
    
    def transform(self, X, y=None):
        X_new = X.copy()
        for column in self.columns:
            if isinstance(X_new[column].dtype, object):
                X_new[column] = X_new[column].astype(str)
            X_new[column] = X_new[column].map(self.__decode_string)
        return X_new


class LowFrequencyFilter(BaseEstimator, TransformerMixin):
    
    def __init__(self, column, min_frequency):
        self.column = column
        self.min_frequency = min_frequency
        
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise FitFailedWarning('X must be an instance of pandas.Dataframe.')
        return self
    
    def transform(self, X, y=None):
        vals, counts = np.unique(X[self.column], return_counts=True)
        mask = np.isin(X[self.column], vals[counts >= self.min_frequency])
        return X[mask]


class WordTokenizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise FitFailedWarning('X must be an instance of pandas.Dataframe.')
        return self

    def __remove_punct(self, tokens):
        if tokens is not None:
            return np.array([t for t in tokens if not all(ch in string.punctuation for ch in t)])

    def transform(self, X, y=None):
        for column in self.columns:
            X[column] = X[column].astype(str)
            X[column] = X[column].str.lower()
            X[column] = X[column].apply(word_tokenize)
            X[column] = X[column].apply(self.__remove_punct)
        return X
    

class StopwordFilter(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.columns = columns
        nltk.download('stopwords', quiet=True)
        self.stopwords = set(stopwords.words('english'))
        
        
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise FitFailedWarning('X must be an instance of pandas.Dataframe.')
        return self
    
    def __remove(self, tokens):
        if tokens is not None:
            return np.array([tok for tok in tokens if tok not in self.stopwords])
    
    def transform(self, X, y=None):
        for column in self.columns:
            X[column] = X[column].apply(self.__remove)
        return X


class WordStemmer(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.columns = columns
        self.stemmer = PorterStemmer()
        
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise FitFailedWarning('X must be an instance of pandas.Dataframe.')
        return self
    
    def __stem(self, tokens):
        if tokens is not None:
            return [self.stemmer.stem(tok) for tok in tokens]
    
    def transform(self, X, y=None):
        for column in self.columns:
            X[column] = X[column].apply(self.__stem)
        return X


class WordTokenJoin(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise FitFailedWarning('X must be an instance of pandas.Dataframe.')
        return self
    
    def __join(self, tokens):
        if tokens is not None:
            return ' '.join(tokens)
    
    def transform(self, X, y=None):
        for column in self.columns:
            X[column] = X[column].apply(self.__join)
        return X
