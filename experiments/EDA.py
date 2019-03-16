# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 21:15:26 2019

@author: zhangka
"""

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["patch.force_edgecolor"] = True
sns.set_style("white")

import pandas as pd
from sklearn.preprocessing import LabelEncoder


# =============================================================================
# EDA insights:
# 1. The data is centered around white male US people. More than 89% from US
# 3. Quantitative variables are not on the same scales, they need to be scaled.
# 4. three Qualitative variables contain missing values: WorkClass, occupation, native-country 
# 5. The variables education and education-num are highly correlated. We will continue with education-num since it is ordered
# 6. Sex and relationship seem to be negatively correlated, but that makes sense (husband = male), (femmale = wife) 

# ============================================================================= 


# Plot a heatmap for the correlation matrix
def plot_heatmap(df, feature, feature_name):
    fig = plt.figure() 
    corr = df[feature].corr() 
    sns.heatmap(corr,cmap="RdBu_r", center=0)
    plt.title(feature_name)
    fig.savefig("exports/" + feature_name + '.png', bbox_inches='tight')
    
# Encode the categorical features as numbers
def number_encode_features(df, qualitative):
    encoders = {}
    qualitative_encoded = []
    for column in qualitative:
        encoders[column] = LabelEncoder()
        df[column + '_E'] = encoders[column].fit_transform(df[column].astype(str)) # TO BE CHECKED
        qualitative_encoded.append(column + '_E')
    return df, qualitative_encoded

# Plot a pointplot
def pointplot(x, y, **kwargs):
    ax=plt.gca()
    ts = pd.DataFrame({'time': x, 'val': y})
    ts = ts.groupby('time').mean()
    ts.plot(ax=ax)

def EDA_basic():
    
    # Start with some basic EDA
    print(X_train.head())
    print(X_train.describe())
    print(X_train.info())
    
    # There are three columns with missing valueshave missing values 
    print(X_train.isnull().sum())
    
    # Take a look at the native country. Most seem to originate from the US
    print((X_train["native-country"].value_counts() / X_train.shape[0]).head())
    
    # Calculate the correlation and plot it
    plot_heatmap(train, quantitative+qualitative_encoded+[y_train.name], 'Correlation_matrix')

    # This shows that the correlations do make sense.
    train[['education', 'education-num']].head(10)
    train[['sex', 'relationship']].head(10)

    #Plot some pointplots wrt the target variable.
    f = pd.melt(train, id_vars=y_train.name, value_vars=quantitative+qualitative_encoded)
    g = sns.FacetGrid(data=f, col='variable', col_wrap=4, size=5, sharex=False, sharey=False)
    g = g.map(pointplot, "value", y_train.name)
    g.savefig("exports/" + 'pointplots.png', bbox_inches='tight')

def EDA_quantitative():

    # Take a look at the distribution plots
    f = pd.melt(X_train, value_vars=quantitative) 
    g = sns.FacetGrid(f, col="variable", hue = 'variable', col_wrap=3, 
                      size = 4, sharex=False, sharey=False,\
                      margin_titles=True)
    g = g.map(sns.distplot, "value", bins= 15)
    g.savefig("exports/" + 'quant_distribution.png', bbox_inches='tight')

    # Capital-gain and capital-loss are very 0 centered
    print(X_train["capital-gain"].value_counts())
    print(X_train["capital-loss"].value_counts())    

def EDA_qualitative():
    
    # Convert to category
    for q in qualitative:
        X_train[q] = X_train[q].astype('category')
        print('The categories for {} are {}'.format(q, X_train[q].cat.categories))
        
    #to pass arguments we need to write a little wrapper
    def countplot(x, y, **kwargs):
        sns.countplot(y=x, hue=y)
        plt.xticks(rotation=45)
    
    f = pd.melt(train, id_vars=y_train.name, value_vars=qualitative) 
    g = sns.FacetGrid(data=f, col='variable', col_wrap=3, size=4, 
                      sharex=False, sharey=False) 
    g = g.map(countplot,'value', y_train.name)    
    plt.legend()
    g.savefig("exports/" + 'qual_distribution.png', bbox_inches='tight')


if __name__ == '__main__':
    
    train = pd.concat([X_train, y_train], axis=1)
    quantitative = [t for t in X_train.columns if X_train.dtypes[t] != 'object']
    qualitative = [t for t in X_train.columns if X_train.dtypes[t] == 'object'] 
    
    train, qualitative_encoded = number_encode_features(train, qualitative)        
    
    EDA_basic()
    EDA_quantitative()
    EDA_qualitative()
    