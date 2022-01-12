# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:34:52 2019

@author: zhangka
"""

from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow import keras


# Create a keras model
def create_model(input_dim, optimizer='adam',dropout=0.2,
                 kernel_initializer='uniform'):
    model=keras.Sequential()
    model.add(keras.layers.Dense(units=49,
                                 activation='relu', 
                                 kernel_initializer=kernel_initializer, 
                                 input_dim=input_dim))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, 
                                 activation='sigmoid',
                                 kernel_initializer=kernel_initializer))
    print(model.summary())
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    return model


def get_model_pipeline(model='logistic_regression'):
    
    if model == 'logistic_regression':
        pipeline_model=LogisticRegression(C=1.0,
                                          penalty='l2',
                                          random_state=42)
    
    else:
        pipeline_model=keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model,
                                                              batch_size=10,
                                                              epochs=50,
                                                              shuffle=True,
                                                              input_dim=97)
    
    return pipeline_model
    
    
    