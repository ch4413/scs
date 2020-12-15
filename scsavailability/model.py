from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from scipy import stats

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
    
    if "Faults" in features or "Totes" in features:
    
        X = df[df.columns[4:]]
        
    elif "Faults" in features:
        
        X = df[df.columns[4:-1]]
        
    elif "Totes" in features:
         
        X = df['TOTES']
        
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

    
def run_RF_model(X_train, X_test, y_train, y_test,RF_options = {'num_trees': 100, 
                                                                'criterion':'mse', 
                                                                'max_depth':None, 
                                                                'dtree':False}):
    
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
    pred: pandas Series
        model predictions for plotting
    
    Example
    --------
    RF_mdl,predictions=run_RF_model(X_train, X_test, y_train, y_test,num_trees=100, criterion = 'mse', max_depth=None, dtree=False):
    
    """

    #set up metrics dataframe
    
    fit_metrics = pd.DataFrame(index = ['MAE','MSE','RMSE','MAPE%','ACC%','OOB','R2_Train','R2_Pred'])

    if RF_options['dtree']==True:

            #Fit decision Tree

            dtree_model = DecisionTreeRegressor(criterion = RF_options['criterion'], max_depth= RF_options['max_depth'])
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
        
    model = RandomForestRegressor(n_estimators=RF_options['num_trees'], 
                                  criterion = RF_options['criterion'], 
                                  max_depth=RF_options['max_depth'],
                                  oob_score = True)
    
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
    
    Importance = pd.DataFrame({'Importance': model.feature_importances_,'Feature':X_train.columns}).sort_values(by='Importance', ascending=False)
    Importance = Importance.reset_index()
    Importance = Importance.drop('index',axis=1)

    #plt.figure(figsize=(20,5))
    #sns.barplot(data = Importance, x= 'Feature', y='Importance', order=Importance[1:11].sort_values('Importance',ascending=False).Feature,color='b')
    #plt.xlabel('Asset',fontsize=18)
    #plt.xticks(fontsize=18)

    print('Feature Importance Ranking: \n \n',Importance.head(10))
    print('\nRegression Metrics: \n \n', fit_metrics,'\n')
    
    return model,pred

def run_XGB_model(X_train, X_test, y_train, y_test,XGB_options = {'num_trees': 100, 
                                                                'max_depth':None}):
    
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
    pred: pandas Series
        model predictions for plotting
    
    Example
    --------
    RF_mdl,predictions=run_RF_model(X_train, X_test, y_train, y_test,num_trees=100, criterion = 'mse', max_depth=None, dtree=False):
    
    """

    #set up metrics dataframe
    
    fit_metrics = pd.DataFrame(index = ['MAE','MSE','RMSE','MAPE%','ACC%','OOB','R2_Train','R2_Pred'])

        
    model = XGBRegressor(n_estimators=XGB_options['num_trees'], max_depth=XGB_options['max_depth'])
    
    model.fit(X_train, y_train)

    #Predicting using random forest

    pred = model.predict(X_test)

    #Fill dataframe with metrics

    mape = np.mean(np.abs((y_test - pred) / np.abs(y_test)))

  

    fit_metrics['XGB Metrics'] = [metrics.mean_absolute_error(y_test, pred),
                                 metrics.mean_squared_error(y_test, pred),
                                 np.sqrt(metrics.mean_squared_error(y_test, pred)),
                                 round(mape * 100, 2),round(100*(1 - mape), 2),
                                 'N/A' ,model.score(X_train,y_train),
                                 model.score(X_test,y_test)]
    
    #Output feature importance
    
    Importance = pd.DataFrame({'Importance': model.feature_importances_,'Feature':X_train.columns}).sort_values(by='Importance', ascending=False)
    Importance = Importance.reset_index()
    Importance = Importance.drop('index',axis=1)

    #plt.figure(figsize=(20,5))
    #sns.barplot(data = Importance, x= 'Feature', y='Importance', order=Importance[1:11].sort_values('Importance',ascending=False).Feature,color='b')
    #plt.xlabel('Asset',fontsize=18)
    #plt.xticks(fontsize=18)

    print('Feature Importance Ranking: \n \n',Importance.head(10))
    print('\nRegression Metrics: \n \n', fit_metrics,'\n')
    
    return model,pred,Importance



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
    pred: pandas Series
        model predictions for plotting    
    
    Example
    --------
    Linear_mdl,predictions=run_LR_model(X_train, X_test, y_train, y_test):
    
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

    Coeff = pd.DataFrame({'Coefficients': model.coef_,'Feature':X_train.columns}).sort_values(by='Coefficients',ascending=False)
    Coeff = Coeff.reset_index()
    Coeff = Coeff.drop('index',axis=1)

    #plt.figure(figsize=(20,5))
    #sns.barplot(data = Coeff, x= 'Feature', y='Coefficients',order=Coeff[:10].sort_values('Coefficients').Feature,color='b')
    #plt.xlabel('Feature',fontsize=18)
    #plt.xticks(fontsize=18)
    
    return model, pred, Coeff, fit_metrics

def select_features(X, y, model, **kwargs):
    
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
    
    X = X.copy()
    y = y.copy()
    
    #Reducing Demensionality

    #Fit select model

    sel = SelectFromModel(estimator = model, **kwargs).fit(X,y)

    #Set selected features

    selected_feat = X.columns[(sel.get_support())]

    print('\nNumber of Selected Features:' + str(len(selected_feat)) ,'\n')

    #Reduce number of features

    X_sel = X[selected_feat]
    
    return X_sel

def cross_validate_r2(model, X, y, n_folds=10, shuffle = True, random_state = None):
    
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
    scores = cross_val_score(model, X, y, scoring='r2', cv=folds)    
    df_cross_val[' R2 Scores'] = np.append(scores,[scores.mean(),scores.std()])
        
    print('\nCross Validation Scores ' + str(model) + ': \n \n', df_cross_val)
    
    return scores.mean(), df_cross_val

def stats_model(lm,X,y):
    
    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X)

    newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b

    print(newX)
    
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]

    sd_b = np.round(sd_b,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilities"] = [params,sd_b,ts_b,p_values]
    print(myDF3)

'-------------------------------------------------------------------------------------------------------------------------------------- '   
    
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
    model, r2_scores = fit_n_r2((X, y, LinearRegression))
    """
    model = model_type(**kwargs)
    folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
    scores = cross_val_score(model, X, y, scoring='r2', cv=folds)

    return model, scores



