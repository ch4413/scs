from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def run_model(df, modeltype = 'RF',random_state = None, num_trees=100, criterion = 'mse',
                                                max_depth=None, dtree=True,select='mean',cv=False):
    '''
    function that runs ML models based off chosen features and selected target variable:
    
    1. performs test train split,
    2. runs decision tree model (if required), 
    3. outputs decision tree visulation (if required),
    4. runs random forest model or linear model
    5. outputs residual distrubution, scatter plot and feature importance/coefficient
    6. runs selected model (if required)
    7. outputs regression metrics from model(s) 
    
    Parameters:
    
    df: input data frame containing features and target variable
    modeltype: Linear or random forest model (RF by default)
    num_trees: number of tree for random forrest model (100 by default)
    dtree: option for running decision tree model (True by default)
    select: option for running selected random forest model (True by default)
    visualise: option for outputing decision tree visualisation (False by default)
    criterion: type of criterion used, either 'mse' or 'mae' ('mse' by default)
    max_depth: maximum depth of the tree (None by default)
    random_state: ensures same split for comparison of results for different models (None by default)
    cv: to run cross validation on model, number indicated number of folds (False by default)
    
    
    Note: preprocessing functions must be run prior to this function to ensure dataframe is formatted correctly
    
    '''
    
    #define target variable and features
    
    X = df.drop(['Datetime','Downtime'],axis=1)
    y = df['Downtime']
    
    #train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    
    #Set up metrics dataframe
    
    if modeltype == 'RF':
    
        fit_metrics = pd.DataFrame(index = ['MAE','MSE','RMSE','MAPE%','ACC%','OOB','R2_Train','R2_Pred'])
    
    if modeltype == 'LM':
        
        fit_metrics = pd.DataFrame(index = ['MAE','MSE','RMSE','MAPE%','ACC%','R2_Train','R2_Pred'])
    
    #import metrics
    
    from sklearn import metrics
    
    if modeltype == 'RF':

        if dtree==True:

            #Fit decision Tree

            from sklearn.tree import DecisionTreeRegressor

            dtree_model = DecisionTreeRegressor(criterion = criterion, max_depth=max_depth)
            dtree_model.fit(X_train,y_train)

            #Predicting using decision tree

            dtree_pred = dtree_model.predict(X_test)

            #Fill dataframe with metrics

            mape = np.mean(np.abs((y_test - dtree_pred) / np.abs(y_test)))

            fit_metrics['D_Tree'] = [metrics.mean_absolute_error(y_test, dtree_pred),metrics.mean_squared_error(y_test, dtree_pred),
                                    np.sqrt(metrics.mean_squared_error(y_test, dtree_pred)),round(mape * 100, 2),round(100*(1 - mape), 2),
                                    'N/A',dtree_model.score(X_train,y_train),dtree_model.score(X_test,y_test)]

        #Fit Model
        
    if modeltype == 'RF':

        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=num_trees,criterion = criterion,max_depth=max_depth,oob_score = True)

    if modeltype == 'LM':    

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()

    model.fit(X_train, y_train)

    #Predicting using random forest

    pred = model.predict(X_test)

    #Fill dataframe with metrics

    mape = np.mean(np.abs((y_test - pred) / np.abs(y_test)))

    if modeltype == 'RF':

        fit_metrics[str(modeltype)] = [metrics.mean_absolute_error(y_test, pred),metrics.mean_squared_error(y_test, pred),
                                np.sqrt(metrics.mean_squared_error(y_test, pred)),round(mape * 100, 2),round(100*(1 - mape), 2),
                                model.oob_score_,model.score(X_train,y_train),model.score(X_test,y_test)]

    if modeltype == 'LM':

        fit_metrics[str(modeltype)] = [metrics.mean_absolute_error(y_test, pred),metrics.mean_squared_error(y_test, pred),
                            np.sqrt(metrics.mean_squared_error(y_test, pred)),round(mape * 100, 2),round(100*(1 - mape), 2),
                            model.score(X_train,y_train),model.score(X_test,y_test)]

    if modeltype == 'RF':

        #Output Feature Importance

        Importance = pd.DataFrame({'Importance': model.feature_importances_,
                                   'Feature':X.columns}).sort_values(by='Importance', ascending=False)
        Importance = Importance.reset_index()
        Importance = Importance.drop('index',axis=1)
        plt.figure(figsize=(20,5))
        sns.barplot(data = Importance, x= 'Feature', y='Importance',
                    order=Importance[:10].sort_values('Importance',ascending=False).Feature)
        plt.xlabel('Feature')
        print('Feature Importance Ranking: \n \n',Importance.head(10))

    if modeltype == 'LM':

        #Output model coefficients

        Coeff = pd.DataFrame({'Coefficients': model.coef_,
                                   'Feature':X.columns}).sort_values(by='Coefficients')
        Coeff = Coeff.reset_index()
        Coeff = Coeff.drop('index',axis=1)
        plt.figure(figsize=(20,5))
        sns.barplot(data = Coeff, x= 'Feature', y='Coefficients',
                    order=Coeff[:10].sort_values('Coefficients').Feature)
        plt.xlabel('Feature')
        print('Feature Coefficient Ranking: \n \n',Coeff.head(10))
            
    if select != False:
    
        #Reducing Demensionality

        #Fit select model

        sel = SelectFromModel(model,threshold=select)
        sel.fit(X_train, y_train)
        
        #Set selected features

        selected_feat= X_train.columns[(sel.get_support())]
        
        print('\nNumber of Selected Features:' + str(len(selected_feat)))

        #Reduce number of features

        X_sel = df[selected_feat]

        #Train Test Split

        X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3,random_state=random_state)

        #Fit reduced model

        model.fit(X_train,y_train)

        #Predicting using random forest

        sel_pred = model.predict(X_test)

        #Fill dataframe with metrics

        mape = np.mean(np.abs((y_test - sel_pred) / np.abs(y_test)))
                  
        if modeltype == 'RF':

            fit_metrics[str(modeltype) + ' Reduced'] = [metrics.mean_absolute_error(y_test, sel_pred),
                                                   metrics.mean_squared_error(y_test, sel_pred),
                                                   np.sqrt(metrics.mean_squared_error(y_test, sel_pred)),
                                                   round(mape * 100, 2),round(100*(1 - mape), 2),
                                                   model.oob_score_,
                                                   model.score(X_train,y_train),
                                                   model.score(X_test,y_test)]
                  
        if modeltype == 'LM':
                  
            fit_metrics[str(modeltype) + ' Reduced'] = [metrics.mean_absolute_error(y_test, sel_pred),
                                                   metrics.mean_squared_error(y_test, sel_pred),
                                                   np.sqrt(metrics.mean_squared_error(y_test, sel_pred)),
                                                   round(mape * 100, 2),round(100*(1 - mape), 2),
                                                   model.score(X_train,y_train),
                                                   model.score(X_test,y_test)]
        
    #print metrics
    
    #print('\nRegression Metrics: \n \n', fit_metrics)
    
    #cross validation
    
    if cv != False:
        
        df_cross_val = pd.DataFrame(index = [str(i) for i in range(1,cv+1)]+['Mean','STD'])
        
        from sklearn.model_selection import cross_val_score
        
        if (dtree == True) and (modeltype == 'RF'):          
                  
            scores = cross_val_score(dtree_model, X, y, cv=cv)
            df_cross_val['D_Tree R2 Scores'] = np.append(scores,[scores.mean(),scores.std()])
        
        scores = cross_val_score(model, X, y, cv=cv)
        df_cross_val[str(modeltype) + ' R2 Scores'] = np.append(scores,[scores.mean(),scores.std()])
        
        scores = cross_val_score(model, X_sel, y, cv=cv)
        df_cross_val[str(modeltype) + ' Reduced R2 Scores'] = np.append(scores,[scores.mean(),scores.std()])
        
        print('\nCross Validation Scores: \n \n', df_cross_val)
    
    return fit_metrics, Importance.head(10)


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



