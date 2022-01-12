# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:42:24 2019

@author: zhangka
"""

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd


from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import CategoricalDtype


class ColumnsSelector(BaseEstimator, TransformerMixin):
    '''Select columns based on a given type
    Args:
        type: (string) type such as e.g: int64, str, 
    '''        
    
    def __init__(self, type):
        self.type = type

    def fit(self, X, y=None):
        return self

    def transform(self,X):
        return X.select_dtypes(include=[self.type])


class CategoricalImputer(BaseEstimator, TransformerMixin):
    '''For categorical columns imputation will be done by chooosing a strategy,
    such as fill most frequent. Since sklearn imputer only works for
    numerical values. Fit will create a dictionary for a category and transform
    will impute.
  
    Args:
        strategy: (string) Default is imputation by most_frequent, 
                 if not then 0 is imputed
                 columns: (list) Provide the list of columns with missing values 
    '''

    def __init__(self, columns = None, strategy='most_frequent'):
        self.columns = columns
        self.strategy = strategy
    
    def fit(self,X, y=None):
        if self.columns is None:
            self.columns = X.columns
            
        if self.strategy is 'most_frequent':
            self.fill = {column: X[column].value_counts().index[0] for 
                         column in self.columns}
        else:
            self.fill = {column: '0' for column in self.columns}
            
        return self
      

    def transform(self,X):
        X_copy = X.copy()
        for column in self.columns:
            X_copy[column] = X_copy[column].fillna(self.fill[column])
        
        return X_copy

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    
    ''' For categorical columns an encoding strategy is needed. Here we will
    choose pd.get_dummies, which is a one-hot encoding based strategy.
    Since we need to try to fit all possible categories, we will concatenate
    our feature data. This to prevent that we would encounter unseen categories.
    For each category will we transform to a corresponding column name with
    category type values.
    
    Args:
         dropfirst: (boolean) True drops the first column, this to prevent 
         multicollinearity. False keeps the first column.
    
    '''
    def __init__(self, dropFirst=True):
        self.categories=dict()
        self.dropFirst=dropFirst

    def fit(self, X, y=None):
        train = X.copy() 
        train = train.select_dtypes(include=['object'])
        for column in train.columns:
            self.categories[column] = train[column].value_counts().index.tolist()
            
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.select_dtypes(include=['object'])
        for column in X_copy.columns:
            X_copy[column] = X_copy[column].astype({column:
                CategoricalDtype(self.categories[column])})
        
        return pd.get_dummies(X_copy, drop_first=self.dropFirst)

# we need a pipeline for the numerical columns and the categorical columns.
def get_preprocessing_pipeline():
    ''' The get_preprocessing pipeline combines the pipeline for numerical
    and the pipeline for categorical features through the FeatureUnion function
    
    Returns:
        The preprocessing pipeline
    
    '''
    pipeline_num = Pipeline([("num_selector", ColumnsSelector(type='int64')),
                                   ("scaler", StandardScaler())])
    
    missing_cols = ['workclass','occupation', 'native-country']
    pipeline_cat = Pipeline([("cat_selector", ColumnsSelector(type='object')),
                             ("cat_imputer", CategoricalImputer(columns=missing_cols)),
                             ("encoder", CategoricalEncoder(dropFirst=True))])

    pipeline_processed = FeatureUnion([("pipeline_num", pipeline_num), 
                ("pipeline_cat", pipeline_cat)])
    
    return pipeline_processed