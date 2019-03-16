# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 23:15:27 2019

@author: zhangka
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.base import BaseEstimator, TransformerMixin 
from pandas.api.types import CategoricalDtype

from sklearn.metrics import accuracy_score, roc_curve, classification_report, confusion_matrix

from sklearn.externals import joblib
import os

# A selector class to select columns of a certain type
class ColumnsSelector(BaseEstimator, TransformerMixin):
  
  def __init__(self, type):
    self.type = type
  
  def fit(self, X, y=None):
    return self
  
  def transform(self,X):
    return X.select_dtypes(include=[self.type])


# An imputer class for the missing categorical values
class CategoricalImputer(BaseEstimator, TransformerMixin):
  
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
      self.fill ={column: '0' for column in self.columns}
      
    return self
      
  def transform(self,X):
    X_copy = X.copy()
    for column in self.columns:
      X_copy[column] = X_copy[column].fillna(self.fill[column])
    return X_copy

# We will use one hot encoding using get_dummies
class CategoricalEncoder(BaseEstimator, TransformerMixin):
  
  def __init__(self, dropFirst=True):
    self.categories=dict()
    self.dropFirst=dropFirst
  

  def fit(self, X, y=None):
    train = pd.concat([X_train, X_val]) # we might need to add the test data to convert as well
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


if __name__ == '__main__':
        
    # We will do a train, validation split.
    X_training, X_val, y_training, y_val = train_test_split(X_train, y_train, 
                                                      train_size=0.75, random_state=42)

    # we need a pipeline for the numerical columns and the categorical columns.
    pipeline_num = Pipeline([("num_selector", ColumnsSelector(type='int64')),
                                   ("scaler", StandardScaler())])
    
    
    pipeline_cat = Pipeline([("cat_selector", ColumnsSelector(type='object')),
                                   ("cat_imputer", CategoricalImputer(columns=['workclass','occupation', 'native-country'])),
                                   ("encoder", CategoricalEncoder(dropFirst=True))])

    pipeline_processed = FeatureUnion([("pipeline_num", pipeline_num), 
                ("pipeline_cat", pipeline_cat)])
    
    
    # From EDA we decided to drop education    
    X_training.drop(['education'], axis=1, inplace=True)
    
    # Create a pipeline of transformers and estimator/
    pipeline_full = Pipeline([('pipeline_processed', pipeline_processed),
                              ('model_lr',LogisticRegression(random_state=42))])
    
    # Run the preprocessing pipeline with transformations followed by fitting the estimator
    X_train_processed=pipeline_full.fit(X_training, y_training)
    
    #Evaluate
    X_val.drop(['education'], axis=1, inplace=True)
    y_val_pred = pipeline_full.predict(X_val)   
    
    score = accuracy_score(y_val_pred, y_val.values) 
    print("baseline LR model validation score: {0:.2f} %".format(100 * score))  

    cfm = confusion_matrix(y_val_pred, y_val.values)
    fig = plt.figure()
    sns.heatmap(cfm, annot=True)
    plt.xlabel('Predicted classes')
    plt.ylabel('Actual classes')
    fig.savefig("exports/" + 'baseline_lr_confusion_matrix.png', bbox_inches='tight')

    
    # Check after cross validation
    scores = cross_val_score(pipeline_full, X_training, 
         y_training, cv=5)
    print("baseline LR model cross-validation score: {0:.2f} %".format(100*np.mean(scores))) # baseline = 0.8496318076374738
    
    # Save to file in the current working directory
    pkl_filename = "baseline_model.pkl"  
    joblib.dump(pipeline_full, os.path.join('experiments/', pkl_filename))

    
    # Load from file
    with open(os.path.join('experiments/', pkl_filename), 'rb') as file:  
        pickle_model = pickle.load(file)
    
    # Load from file
    joblib_model = joblib.load(joblib_file)
    
    
    
    # Calculate the accuracy score and predict target values
    score = pickle_model.score(X_val, y_val)  
    print("Test score: {0:.2f} %".format(100 * score))  
    Ypredict = pickle_model.predict(X_val)  