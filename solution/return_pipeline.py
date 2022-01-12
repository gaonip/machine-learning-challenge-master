# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 23:15:27 2019

@author: zhangka
"""

import os

from solution.model import get_model_pipeline
from solution.preprocessing import get_preprocessing_pipeline

from sklearn.pipeline import Pipeline
from sklearn.externals import joblib


# Save to file in the current working directory
def save_model_to_pkl(pipeline, model='model'):
    
    pkl_filename = model + "_model.pkl"   
    joblib.dump(pipeline, os.path.join('experiments/', pkl_filename))

# Load from file
def get_model_from_pkl(pkl_filename):
    
    joblib_model = joblib.load(os.path.join('experiments/', pkl_filename))
    return joblib_model

# Save the NN
def save_pipeline_keras(pipeline,folder_name="experiments"):
    os.makedirs(folder_name, exist_ok=True)
    joblib.dump(pipeline.named_steps['pipeline_processed'], open(folder_name+'/'+'pipeline_processed.pkl','wb'))
    joblib.dump(pipeline.named_steps['pipeline_model'], open(folder_name+'/'+'pipeline_model.pkl','wb'))
    #pipeline.named_steps['pipeline_model'].pipeline.save(folder_name+'/lstm.h5')


# Function to return the pipeline
def get_pipeline(model='logistic_regression'):
    
    ''' The get_pipeline function assembles the preprocessing pipeline 
    with the model pipeline. Depending if you pass it the 'neural_net' 
    or the 'logistic_regresion' a different classifier will be used in
    the pipeline.
    
    Args:
        model: (string): neural_net to return a keras neural net, either
                         logistic_regression to return a logistic regresion.
    
    Return: The full pipeline
    '''
    
    pipeline_processed = get_preprocessing_pipeline()
    pipeline_model = get_model_pipeline(model)
    
    pipeline_full = Pipeline([('pipeline_processed', pipeline_processed),
                              ('pipeline_model', pipeline_model)],
                                memory="/tmp")
    
    # comment this out, if you don't want to save the model
    save_model_to_pkl(pipeline_full, model)
    #save_pipeline_keras(pipeline_full)
    
    return pipeline_full 
