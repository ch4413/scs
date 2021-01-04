from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scipy import stats
import statsmodels.api as sm

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def gen_feat_var(df,target = "Downtime",features = ["Faults","Totes"]):
    
    """
    Summary
    -------
    Splits dataframe into features and target variable
    ----------
    df: pandas DataFrame
        dataframe of features and target variables

    Returns
    -------
    X: pandas DataFrame
        dataframe of features
    Y: pandas Series
        series of target variables
    Example
    --------
    X, y = gen_feat_var(df)
    """
    
    #define target variable and features
    
    df = df[~df[target].isnull()]
    
    if "Faults" in features and "Totes" in features:
    
        X = df[df.columns[4:]]
        
    elif "Faults" in features:
        
        X = df[df.columns[4:-1]]
        
    elif "Totes" in features:
         
        X = pd.DataFrame(df['TOTES'])
        
    else:
        
        print('Features not valid')
        
    y = df[target]
    
    return X,y
    
    
def split(X,y,split_options = {'test_size':0.3,
                               'random_state':None}):
    
    """
    Summary
    -------
    Performs test train split on features and target variable
    ----------
    X: pandas DataFrame
        dataframe of features
    Y: pandas Series
        series of target variables
    test_size: Numeric
        proportion of data in test
    random_state: Integer
        sets seed if required
    
    Returns
    -------
    X_train: pandas DataFrame
        dataframe of training features
    X_test: pandas Series
        dataframe of test features
    y_train: pandas Series
        series of training target variables
    y_test: pandas Series
        series of test target variables    
        
    Example
    --------
    X_train, X_test, y_train, y_test = split(X,y,test_size=0.3,random_state=101)
    """
    
    #train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_options['test_size'], random_state=split_options['random_state'])
    
    return X_train, X_test, y_train, y_test

def run_LR_model(X_train, X_test, y_train, y_test, **kwargs):
    
    """
    Summary
    -------
    Runs linear regresion model and outputs regression metrics and feature coefficients
    ----------
    X_train: pandas DataFrame
        dataframe of training features
    X_test: pandas Series
        dataframe of test features
    y_train: pandas Series
        series of training target variables
    y_test: pandas Series
        series of test target variables    
    
    Returns
    -------
    
    model: sklearn model object
        fitted linear regression model
    pred: pandas Series
        model predictions for plotting    
    
    Example
    --------
    Linear_mdl,predictions=run_LR_model(X_train, X_test, y_train, y_test):
    
    """
    
    #set up metrics dataframe
    
    fit_metrics = pd.DataFrame(index = ['MAE','MSE','RMSE','MAPE%','ACC%','R2_Train','R2_Pred'])
    
    #Fit Model
    
    model = LinearRegression(**kwargs)
    
    model.fit(X_train, y_train)

    #Predicting using model

    pred = model.predict(X_test)

    #Fill dataframe with metrics

    mape = np.mean(np.abs((y_test - pred) / np.abs(y_test)))

    fit_metrics['LM Metrics'] = [metrics.mean_absolute_error(y_test, pred),
                                 metrics.mean_squared_error(y_test, pred),
                                 np.sqrt(metrics.mean_squared_error(y_test, pred)),
                                 round(mape * 100, 2),
                                 round(100*(1 - mape), 2),
                                 model.score(X_train,y_train),
                                 model.score(X_test,y_test)]
    
    #Output model coefficients

    Coeff = pd.DataFrame({'Coefficients': model.coef_,'Feature':X_train.columns}).sort_values(by='Coefficients',ascending=False)
    Coeff = Coeff.reset_index()
    Coeff = Coeff.drop('index',axis=1)

    #plt.figure(figsize=(20,5))
    #sns.barplot(data = Coeff, x= 'Feature', y='Coefficients',order=Coeff[:10].sort_values('Coefficients').Feature,color='b')
    #plt.xlabel('Feature',fontsize=18)
    #plt.xticks(fontsize=18)
    
    return model, pred, Coeff, fit_metrics
    

def cross_validate_r2(model, X, y, n_folds=5, shuffle = True, random_state = None):
    
    """
    Summary
    -------
    Generates cross-validated R2 scores for a given number of folds
    ----------
    
    model: sklearn model object
        fitted linear regression model
    X: pandas DataFrame
        dataframe of features
    Y: pandas Series
        series of target variables
    n_folds: integer
        number of splits of the data
    shuffle: Boolean
        choose to shuffle data before splitting
    random_state: Integer
        sets seed if required
    
    Returns
    -------
    
    scores.mean(): float
        mean of R2 scores
    
    Example
    --------
    cv_R2 = cross_validate_r2(model, X, y, n_folds = 10, shuffle = True, random_state = 101)
    
    """

    df_cross_val = pd.DataFrame(index = [str(i) for i in range(1,n_folds+1)]+['Mean','STD'])       
     
    folds = KFold(n_splits = n_folds, shuffle = shuffle, random_state = random_state)
    scores = cross_val_score(estimator = model,X = X,y = y, scoring='r2', cv=folds)    
    df_cross_val[' R2 Scores'] = np.append(scores,[scores.mean(),scores.std()])
        
    print('\nCross Validation Scores: \n \n', df_cross_val)
    
    return scores.mean()

def find_features(X_train, y_train, n):
    X_train = X_train.copy()
    
    max_p = 1
    while max_p > 0.1:
        model = sm.OLS(y_train, X_train)
        results = model.fit()
        top_n = results.pvalues.sort_values(ascending=False).head(n)
        max_p = top_n.tail(1).values[0]
        rm_col = list(results.pvalues.sort_values(ascending=False).head(n).index)
        X_train = X_train.drop(rm_col, axis=1)
    return X_train.columns

def run_OLS(X_train,y_train,X_test,y_test, n):

    Linear_mdl,pred, Coeff, fit_metrics = run_LR_model(X_train, X_test, y_train, y_test) #fit_intercept=False)

    keep_features = find_features(X_train = X_train , y_train=y_train, n=n)

    model = sm.OLS(y_train, X_train[keep_features])
    results = model.fit()

    cv_R2 = cross_validate_r2(model = Linear_mdl, X = X_train[keep_features], y = y_train)

    print(len(keep_features))
    print(results.summary())

    negs = results.params[results.params < 0]
    Coefficients = pd.DataFrame(negs, columns=['Coefficient']).reset_index()

    Linear_mdl,pred, Coeff, fit_metrics = run_LR_model(X_train[keep_features], X_test[keep_features], y_train, y_test) #fit_intercept=False)
    
    R2_OOS = Linear_mdl.score(X_test[keep_features],y_test)

    return cv_R2,R2_OOS,Coefficients