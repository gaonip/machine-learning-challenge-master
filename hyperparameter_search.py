import joblib
import os
import pandas as pd
import numpy as np

import sklearn.metrics
import sklearn.model_selection
from sklearn.model_selection import train_test_split

import sklearn.pipeline
import sys

import matplotlib.pyplot as plt
import seaborn as sns

# Cache the train and test data in {repo}/__data__.
cachedir = os.path.join(sys.path[0], '__data__')
memory = joblib.Memory(cachedir=cachedir, verbose=0)


@memory.cache()
def get_data(subset='train'):
    # Construct the data URL.
    csv_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/'
    csv_url += f'adult/adult.{"data" if subset == "train" else "test"}'
    # Define the column names.
    names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'earns_over_50K']
    # Read the CSV.
    print(f'Downloading {subset} dataset to __data__/ ...')
    df = pd.read_csv(
        csv_url,
        sep=', ',
        names=names,
        skiprows=int(subset == 'test'),
        na_values='?')
    # Split into feature matrix X and labels y.
    df.earns_over_50K = df.earns_over_50K.str.contains('>').astype(int)
    X, y = df.drop(['earns_over_50K'], axis=1), df.earns_over_50K
    return X, y


def score_solution():
    # Ask the solution for the model pipeline.
    # import solution
    from solution.return_pipeline import get_pipeline
    #pipeline = solution.get_pipeline()
    pipeline = get_pipeline(model='logistic_regression')
    error_message = 'Your `solution.get_pipeline` implementation should ' \
        'return an `sklearn.pipeline.Pipeline`.'
    assert isinstance(pipeline, sklearn.pipeline.Pipeline), error_message
    # Train the model on the training DataFrame.
    X_train, y_train = get_data(subset='train')
    pipeline.fit(X_train, y_train)
    # Apply the model to the test DataFrame.
    X_test, y_test = get_data(subset='test')
    y_pred = pipeline.predict_proba(X_test)
    # Check that the predicted probabilities have an sklearn-compatible shape.
    assert (y_pred.ndim == 1) or \
        (y_pred.ndim == 2 and y_pred.shape[1] == 2), \
        'The predicted probabilities should match sklearn''s ' \
        '`predict_proba` output shape.'
    y_pred = y_pred if y_pred.ndim == 1 else y_pred[:, 1]
    # Evaluate the predictions with the AUC of the ROC curve.
    return sklearn.metrics.roc_auc_score(y_test, y_pred)


# Plot the confusion matrix and export as png file
def plot_CFM(y_test, y_pred, filename):
    fig = plt.figure()
    cfm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(cfm, annot=True)
    plt.xlabel('Predicted classes')
    plt.ylabel('Actual classes')
    fig.savefig("exports/" + filename, bbox_inches='tight')

# Plot the ROC curve and export as png file
def plot_ROC(y_test, y_pred, roc_auc, filename):
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y_test.values, y_pred)
    
    fig = plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = "{0:.2f} %".format(100*roc_auc))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    fig.savefig("exports/" + filename, bbox_inches='tight')


def hyperparameter_search():
    from solution.return_pipeline import get_pipeline
    pipeline = get_pipeline(model='logistic_regression')
    error_message = 'Your `solution.get_pipeline` implementation should ' \
        'return an `sklearn.pipeline.Pipeline`.'
    assert isinstance(pipeline, sklearn.pipeline.Pipeline), error_message

    # First run a baseline metrics check.
    X_train, y_train = get_data(subset='train')
    # We will do a train, validation split.
    X_training, X_val, y_training, y_val = train_test_split(X_train, 
                                                            y_train, 
                                                            train_size=0.75, 
                                                            random_state=42)
    pipeline.fit(X_training, y_training)
    y_val_pred = pipeline.predict(X_val)
    
    score = sklearn.metrics.accuracy_score( y_val.values, y_val_pred) 
    print("The model validation score: {0:.2f} %".format(100*score))  

    # Plot the confusion matrix and the ROC curve
    plot_CFM(y_val.values, y_val_pred, 'baseline_lr_confusion_matrix.png')
    
    roc_auc = sklearn.metrics.roc_auc_score( y_val.values, y_val_pred)
    print("The model AUC score: {0:.2f} %".format(100*roc_auc))  

    plot_ROC(y_val.values, y_val_pred, roc_auc, 'baseline_lr_ROC.png')
    
    # Check what cross validation score gives. Not possible yet for neural net.
    kfold = sklearn.model_selection.KFold(n_splits=10, 
                                          shuffle=True, 
                                          random_state=42)
    
    scores = sklearn.metrics.cross_val_score(pipeline, 
                                             X_training, 
                                             y_training, 
                                             cv=kfold) 
    print("The model CV score: {0:.2f} %".format(100*np.mean(scores)))  
    
    #  Lets use gridsearch to tune the hyperparameters
    param_range_fl = [1.0, 0.5, 0.1]

    grid_params_lr = [{'model_lr__penalty':['l1', 'l2'],
                'model_lr__C':param_range_fl}]
    
    gs_lr = sklearn.model_selection.GridSearchCV(estimator=pipeline,
                  param_grid=grid_params_lr,
                  scoring='roc_auc',
                  cv=10)

    # Best params
    print('\nbest params: \n', gs_lr.best_params_)
    #Best score on training data
    print('Best training accuracy: %.3f' %gs_lr.best_score_)    
    
    
    #  Lets use gridsearch to tune the hyperparameters for the NN.
    grid_params_nn = [{'model_keras__batch_size': [25, 32],
              'model_keras__epochs': [40, 100]}]
    
    gs_nn = sklearn.model_selection.GridSearchCV(estimator=pipeline,
                                                 param_grid=grid_params_nn,
                                                 scoring='roc_auc',
                                                 cv=10)
    
    gs_nn.fit(X_training, y_training)   

if __name__ == '__main__':
    print(score_solution())
        
   

    
