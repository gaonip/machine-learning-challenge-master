# Machine Learning challenge

## Some notes in addition to the readme.md

- The environment.yml file is slighty adapted to download some additional plotting libraries sucha as matplotlib as well as tensorflow is installed from conda instead of pip.
- By default the model runs a logistic_regression, if you would like to run a neural net go to challenge.py, go to score_solution() > get_pipeline(model) and change the model parameter to 'neural_net'.
- The solution repository consists of four files (__init__py, model.py, preprocessing.py and return_pipeline.py). The get_pipeline() function is defined in return_pipeline()
- The experiments repository contains model pkl exports, an EDA.py file where exploratory data analysis was performed and the initial baseline tested models.
- The exports repository contains insightful figures from the performed EDA and the initial baseline models.
- The search_hyperparameter.py file is a testing file, used to perform optimization and tuning such as gridsearch. It is not a part of the solution, but can be further explored for future improvements. It is not a full refactored script.
- Finally a lot of improvements are still do be explored. This is only the beginnning, a few possibilities are: 
    - feature engineering, 
    - Model optimization
    - tf.estimator usage, 
    - Pretrained embedding layer as preprocessing step, 
    - model parameters should not be left in te code but inside a params.json 
    - missing values could be imputed with machine learning algorithms such MICE or KNN. ( to be checked )
    - Better packaging, refactoring, testing, logging