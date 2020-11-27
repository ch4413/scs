from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def gen_feat_var(df)
    
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
    
    X = df.drop(['Datetime','Downtime'],axis=1)
    y = df['Downtime']
    
    return X,y
    
    
def split(X,y,test_size=None,random_state=None):
    
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

    
def run_RF_model(X_train, X_test, y_train, y_test,num_trees=100, criterion = 'mse', max_depth=None, dtree=False):
    
    """
    Summary
    -------
    Runs random forest model and decision tree if chosen and outputs regression metrics and feature importance
    ----------
    X_train: pandas DataFrame
        dataframe of training features
    X_test: pandas Series
        dataframe of test features
    y_train: pandas Series
        series of training target variables
    y_test: pandas Series
        series of test target variables    
    num_trees: integer
        number of trees to use in RF model
    criterion: string 'mse'/'mae'
        criterion use to split nodes
    max_depth: integer
        maximum depth of trees
    dtree: boolean
        generate decision tree option
    
    Returns
    -------
    
    model: sklearn model object
        fitted RF model
    
    Example
    --------
    RF_mdl=run_RF_model(X_train, X_test, y_train, y_test,num_trees=100, criterion = 'mse', max_depth=None, dtree=False):
    
    """

    #set up metrics dataframe
    
    fit_metrics = pd.DataFrame(index = ['MAE','MSE','RMSE','MAPE%','ACC%','OOB','R2_Train','R2_Pred'])

    if dtree==True:

            #Fit decision Tree

            dtree_model = DecisionTreeRegressor(criterion = criterion, max_depth=max_depth)
            dtree_model.fit(X_train,y_train)

            #Predicting using decision tree

            dtree_pred = dtree_model.predict(X_test)

            #Fill dataframe with metrics

            mape = np.mean(np.abs((y_test - dtree_pred) / np.abs(y_test)))

            fit_metrics['D_Tree Metrics'] = [metrics.mean_absolute_error(y_test, dtree_pred),
                                             metrics.mean_squared_error(y_test, dtree_pred),
                                             np.sqrt(metrics.mean_squared_error(y_test, dtree_pred)),
                                             round(mape * 100, 2),
                                             round(100*(1 - mape), 2), 'N/A', 
                                             dtree_model.score(X_train,y_train),
                                             dtree_model.score(X_test,y_test)]

    #Fit Model
        
    model = RandomForestRegressor(n_estimators=num_trees,criterion = criterion,max_depth=max_depth,oob_score = True)
    
    model.fit(X_train, y_train)

    #Predicting using random forest

    pred = model.predict(X_test)

    #Fill dataframe with metrics

    mape = np.mean(np.abs((y_test - pred) / np.abs(y_test)))

  

    fit_metrics['RF Metrics'] = [metrics.mean_absolute_error(y_test, pred),
                                 metrics.mean_squared_error(y_test, pred),
                                 np.sqrt(metrics.mean_squared_error(y_test, pred)),
                                 round(mape * 100, 2),round(100*(1 - mape), 2),
                                 model.oob_score_,model.score(X_train,y_train),
                                 model.score(X_test,y_test)]
    
    #Output feature importance
    
    Importance = pd.DataFrame({'Importance': model.feature_importances_,'Feature':X.columns}).sort_values(by='Importance', ascending=False)
    Importance = Importance.reset_index()
    Importance = Importance.drop('index',axis=1)
    plt.figure(figsize=(20,5))
    sns.barplot(data = Importance, x= 'Feature', y='Importance', order=Importance[:10].sort_values('Importance',ascending=False).Feature)
    plt.xlabel('Feature')
    print('Feature Importance Ranking: \n \n',Importance.head(10))

    return model

def run_LR_model(X_train, X_test, y_train, y_test):
    
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
    
    Example
    --------
    Linear_mdl=run_LR_model(X_train, X_test, y_train, y_test):
    
    """
    
    #set up metrics dataframe
    
    fit_metrics = pd.DataFrame(index = ['MAE','MSE','RMSE','MAPE%','ACC%','R2_Train','R2_Pred'])
    
    #Fit Model
    
    model = LinearRegression()
    
    model.fit(X_train, y_train)

    #Predicting using random forest

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

    Coeff = pd.DataFrame({'Coefficients': model.coef_,'Feature':X.columns}).sort_values(by='Coefficients')
    Coeff = Coeff.reset_index()
    Coeff = Coeff.drop('index',axis=1)
    plt.figure(figsize=(20,5))
    sns.barplot(data = Coeff, x= 'Feature', y='Coefficients',order=Coeff[:10].sort_values('Coefficients').Feature)
    plt.xlabel('Feature')
    print('Feature Coefficient Ranking: \n \n',Coeff.head(10))
    
    return model

def select_features(X, X_train, y_train, model, thres = None):
    
    """
    Summary
    -------
    Selects most important features and returns selected features dataframe
    ----------
    X: pandas DataFrame
        dataframe of features
    X_train: pandas DataFrame
        dataframe of training features
    y_train: pandas Series
        series of training target variables
    model: sklearn model object
        fitted linear regression model
    thres: string 'mean'/'median'
        threshold criterion for selecting features
    
    Returns
    -------
    
    X_sel: pandas DataFrame
        dataframe of selected features
    
    Example
    --------
    X_sel=select_features(X, X_train, y_train, model, thres = 'median')
    
    """
    
    #Reducing Demensionality

    #Fit select model

    sel = SelectFromModel(model,threshold=thres)
    sel.fit(X_train, y_train)

    #Set selected features

    selected_feat= X_train.columns[(sel.get_support())]

    print('\nNumber of Selected Features:' + str(len(selected_feat)))

    #Reduce number of features

    X_sel = X[selected_feat]
    
    return X_sel

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

    df_cross_val = pd.DataFrame(index = [str(i) for i in range(1,folds+1)]+['Mean','STD'])       
     
    folds = KFold(n_splits = n_folds, shuffle = shuffle, random_state = random_state)
    scores = cross_val_score(model, X, Y, scoring='r2', cv=folds)    
    df_cross_val[' R2 Scores'] = np.append(scores,[scores.mean(),scores.std()])
        
    print('\nCross Validation Scores: \n \n', df_cross_val)
    
    return scores.mean()

--------------------------------------------------------------------------------------------------------------------------------------    
    
def fit_n_r2(X, Y, model_type, **kwargs):
    """
    Summary
    -------
    Takes variables and fits model with arguments. Return model object.
    Parameters
    ----------
    X: pandas DataFrame
        dataframe of features
    Y: pandas Series
        series of target variables
    model_type: ModelClass
        model that we wish to test

    Returns
    -------
    model: ModelClass
        fitted model
    scores: list
        list of r2 values from cross-validation
    Example
    --------
    model, r2_scores = fit_n_r2((X, Y, LinearRegression))
    """
    model = model_type(**kwargs)
    folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
    scores = cross_val_score(model, X, Y, scoring='r2', cv=folds)

    return model, scores



