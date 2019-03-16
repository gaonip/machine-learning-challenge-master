# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 23:15:27 2019

@author: zhangka
"""

#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from pandas.api.types import CategoricalDtype

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.externals import joblib
import os

import tensorflow as tf
from tensorflow import keras


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
        train = X.copy() # TODO might need to add test data
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

# ToDO
#class KerasWrapper(BaseEstimator, ClassifierMixin):
#    """
#    """
#    def __init__(self, optimizer='adam', dropout=0.2, kernel_initializer='uniform'):
#        self.dropout = dropout
#        self.optimizer = optimizer
#        self.kernel_initializer = kernel_initializer
#        
#    def fit(self, X, y=None):
#        model = keras.Sequential()
#        model.add(keras.layers.Dense(units=64,activation='relu', 
#                                     kernel_initializer=self.kernel_initializer, 
#                                     input_dim=97))
#        model.add(keras.layers.Dropout(self.dropout))
#        model.add(keras.layers.Dense(1, activation='sigmoid',
#                                     kernel_initializer = self.kernel_initializer))
#        model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['acc'])
#        return self
    

# Plot the confusion matrix and export as png file
#def plot_CFM(y_test, y_pred, filename):
#    fig = plt.figure()
#    cfm = confusion_matrix(y_test, y_pred)
#    sns.heatmap(cfm, annot=True)
#    plt.xlabel('Predicted classes')
#    plt.ylabel('Actual classes')
#    fig.savefig("exports/" + filename, bbox_inches='tight')

# Plot the ROC curve and export as png file
def plot_ROC(y_test, y_pred, roc_auc, filename):
    fpr, tpr, threshold = roc_curve(y_val.values, y_val_pred)
    
    fig = plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = "{0:.2f} %".format(100 * roc_auc))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    fig.savefig("exports/" + filename, bbox_inches='tight')
    
    
# Function to return the pipeline
def get_pipeline():
    # we need a pipeline for the numerical columns and the categorical columns.
    pipeline_num = Pipeline([("num_selector", ColumnsSelector(type='int64')),
                                   ("scaler", StandardScaler())])
    
    missing_cols = ['workclass','occupation', 'native-country']
    pipeline_cat = Pipeline([("cat_selector", ColumnsSelector(type='object')),
                             ("cat_imputer", CategoricalImputer(columns=missing_cols)),
                             ("encoder", CategoricalEncoder(dropFirst=True))])

    pipeline_processed = FeatureUnion([("pipeline_num", pipeline_num), 
                ("pipeline_cat", pipeline_cat)])
    
    # Create a pipeline of transformers and estimator/
    pipeline_full = Pipeline([('pipeline_processed', pipeline_processed),
                              ('model_keras', keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model,
                                                              batch_size=10,
                                                              epochs=50))],
                                memory="/tmp") # ('model_keras', model_keras)
    
    return pipeline_full
    

# Create a keras model
def create_model(optimizer='adam',dropout=0.2, kernel_initializer='uniform'):
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=64,activation='relu', 
                                 kernel_initializer=kernel_initializer, 
                                 input_dim=97))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation='sigmoid',
                                 kernel_initializer = kernel_initializer))
    print(model.summary())
    model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['acc'])
    return model
    

if __name__ == '__main__':
        
    # We will do a train, validation split.
    X_training, X_val, y_training, y_val = train_test_split(X_train, y_train, 
                                                      train_size=0.75, random_state=42)

    # Get the pipeline
    pipeline_full = get_pipeline()

    #Run the preprocessing with transformations followed by fitting the estimator
    pipeline_full.fit(X_training, y_training)    
    
    # For the evaluation set
    y_val_pred = pipeline_full.predict(X_val)
    
    score = accuracy_score( y_val.values, y_val_pred) 
    print("baseline LR model validation score: {0:.2f} %".format(100 * score))  

    # Plot the confusion matrix and the ROC curve
    plot_CFM(y_val.values, y_val_pred, 'baseline_lr_confusion_matrix.png')
   
    roc_auc = roc_auc_score( y_val.values, y_val_pred)
    plot_ROC(y_val.values, y_val_pred, roc_auc, 'baseline_lr_ROC.png')
    print("baseline AUC score: {0:.2f} %".format(100 * roc_auc))  

    # Check what cross validation score gives.
    scores = cross_val_score(pipeline_full, X_training, y_training, cv=5) # TODO doesnt work on neural net
    print("baseline LR model cross-validation score: {0:.2f} %".format(100*np.mean(scores))) 
    
    #  Lets use gridsearch to tune the hyperparameters

    
    grid_params_nn = [{'model_keras__batch_size': [25, 32],
              'model_keras__epochs': [40, 100]}]
    
    
    gs_nn = GridSearchCV(estimator = pipeline_full,
                       param_grid = grid_params_nn,
                       scoring = 'roc_auc',
                       cv = 10)
    
    gs_nn.fit(X_training, y_training)    

    
    # Best params
    print('\nbest params: \n', gs_lr.best_params_)
    #Best score on training data
    print('Best training accuracy: %.3f' %gs_lr.best_score_)
    
    # Save to file in the current working directory
    pkl_filename = "baseline_model.pkl"  
    joblib.dump(pipeline_full, os.path.join('experiments/', pkl_filename))
    
    # Load from file
    joblib_model = joblib.load(os.path.join('experiments/', pkl_filename))
    
    # Calculate the accuracy score and predict target values
    score = joblib_model.score(X_val, y_val)  
    print("Test score: {0:.2f} %".format(100 * score))  
    Ypredict = joblib_model.predict(X_val)  