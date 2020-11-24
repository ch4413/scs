def pre_process_av_and_fa_oct_nov(av,fa_oct,fa_nov,
                                  remove_same_location_faults = True):
    '''function that pre-processes the raw csv files:
    1. converts date columns to datetime objects,
    2. assigns quandrants by PLC location, 
    3. drops rows with missing values, 
    4. drops faults that happen in same location at same time (keeps fault with max duration) - if selected 
    
    Note: function will need to be adapated to preprocess availability data from other months
    '''
    
    #convert to percentage downtime
    
    av['Availability'] = 1 - av['Availability']
    av.rename(columns = {'Availability':'Downtime'},inplace = True)

    fa_nov.rename(columns = {'Entry date stamp':'Entry time'},inplace = True)
    fa_nov.rename(columns = {3:'Entry time'},inplace = True)

    fa = fa_nov.append(fa_oct,ignore_index=True)

    #Assign PLC code to Quadrants
    Quad_1 = ['C0' + str(i) for i in range(5,8)]  + ['SCSM0' + str(i) for i in range(1,6)]
    Quad_2 = ['C0' + str(i) for i in range(8,10)] + ['SCSM0' + str(i) for i in range(7,10)] + ['SCSM11']
    Quad_3 = ['C'  + str(i) for i in range(10,13)] + ['SCSM' + str(i) for i in range(11,16)]
    Quad_4 = ['C'  + str(i) for i in range(13,15)] + ['SCSM' + str(i) for i in range(17,21)]


    #Assing faults to Quadrants  
    Quad = []

    for i in fa['PLC']:
        if i in Quad_1:
            Quad.append(1)
        elif i in Quad_2:
            Quad.append(2)
        elif i in Quad_3:
            Quad.append(3)
        elif i in Quad_4:
            Quad.append(4)
        else:
            Quad.append(0)
    fa['Quadrant']=Quad

    #Assign Pick Station to Quadrant
    Quad_1 = ['PTT011','PTT012','PTT021','PTT022','PTT031','PTT032','PTT041','PTT042','PTT051','PTT052']
    Quad_2 = ['PTT071','PTT072','PTT081','PTT082','PTT091','PTT092','PTT101','PTT102']
    Quad_3 = ['PTT111','PTT112','PTT121','PTT122','PTT131','PTT132','PTT141','PTT142','PTT151','PTT152']
    Quad_4 = ['PTT171','PTT172','PTT181','PTT182','PTT191','PTT192','PTT201','PTT202']


    #Assing availability to Quadrants  
    Quad = []
    for i in av['Pick Station']:
        if i in Quad_1:
            Quad.append(1)
        elif i in Quad_2:
            Quad.append(2)
        elif i in Quad_3:
            Quad.append(3)
        elif i in Quad_4:
            Quad.append(4)
        else:
            Quad.append(0)
    av['Quadrant'] = Quad
    
    print('Quadrants Assigned')
    
    fa['Entry time'] = pd.to_datetime(fa['Entry time'],dayfirst=True)
    av['Datetime'] = pd.to_datetime(av['Datetime'],dayfirst=True)
    
    #drop rows where there is no duration data
    fa = fa.dropna(subset = ['Duration'])

    #convert duration string to time delta and then to seconds (float)
    fa['Duration'] = pd.to_timedelta(fa['Duration'].str.slice(start=2))
    fa['Duration'] = fa['Duration'].dt.total_seconds()
    
    #drop faults that happen at same time and in same location (keep only the one with max duration)
    if remove_same_location_faults == True:
        fa = fa.sort_values('Duration').drop_duplicates(subset=['Entry time', 'PLC', 'Desk'],keep='last')
        print('duplicated location faults removed - max duration kept')
        
    dfs = {'faults':fa,"availability":av}
    
    print("Fault and availability data pre-processed")
    return(dfs)
    
    
def floor_time(df,time_col,units = 'H'):
    ''' function that floors datetime column in a pandas df to the specific time unit
    Note: defaults to HOUR'''
    df[time_col] = df[time_col].dt.floor(units)
    print("Dates floored")
    return(df)


def faults_aggregate_and_pivot(df,time_col,fault_level,agg_col,agg_type,break_durations = False, quadrant=None):
    '''function that aggregates fault data by specified metric (agg_type) and quadrant.
    - The quadrant parameter is used in case you want to filter for a specifc quadrant
    (e.g., if you only want observations from quadrant 1)
    - agg type is count/mean/sum
    -fault_level is 'fault ID type' or 'Number' or 'PLC' or 'Quadrant'
    -agg_col should be duration
    -break_durations if you want to bucket up durations in short, medium and long
    '''
    
    
    if break_durations == True:
        df['Duration_category'] = df.groupby(fault_level)["Duration"].apply(lambda x: pd.cut(x, 3,labels=['short','medium','long']))
        cols = [fault_level,'Duration_category']
        df['period'] = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        df = df.groupby([time_col,'Quadrant','period'],as_index = False).agg({agg_col:agg_type})
        if quadrant != None:
            df = df[df['Quadrant'].isin([quadrant, 0])]
        df = pd.pivot_table(df,values = agg_col,index = time_col,columns = ['period'],fill_value=0)
        print('Duration broken into categories Short, Medium, Long')
    else:
        if fault_level == 'Quadrant':
            df = df.groupby([time_col,'Quadrant'],as_index = False).agg({agg_col:agg_type})
        else:
            df = df.groupby([time_col,'Quadrant',fault_level],as_index = False).agg({agg_col:agg_type})
        if quadrant != None:
            df = df[df['Quadrant'].isin([quadrant, 0])]  
        df = pd.pivot_table(df,values = agg_col,index = time_col,columns = fault_level,fill_value=0)
        
    print('Faults aggregated and pivoted')
    return(df)


def availability_quadrant_mean(df,time_col,quadrant=None):
    '''function to aggregate availability at the quadrant level'''
    
    if quadrant != None:
        print("Output will contain data only for Quadrant:" + str(quadrant))
        df = df[df['Quadrant'].isin([quadrant, 0])] 
    df = df.groupby([time_col],as_index=False).agg('mean')
    df = df.drop(['Quadrant'],axis=1)
    df = df.set_index(time_col)
    print('Availability data aggregated by quadrant')
    return(df)

def weight_hours(df,weights = [1,0.5,0.2]):
    
    '''function to include weighted fault data from previous hours
    
    Parameters:
    
    df: input data frame
    weights: weights for hours with first element in array being weight for current hour, second the previous hour etc.
    
    '''
    
    df_weight = pd.DataFrame(data=np.zeros((len(df)-2,len(df.columns))),index=df.index[2:],columns = df.columns)
    
    for i in range(len(weights)-1,len(df)):
        
        for x in range(len(weights)):
         
            df_weight.iloc[i-2] = df_weight.iloc[i-2] + df.iloc[i-x]*weights[x]

    print('Previous Hours Weighted')
    return(df_weight)

def merge_av_fa(av_df,fa_df,min_date=None,max_date=None):
    '''function that merges availability and fault datasets by date index'''
    if min_date is None:
        min_date = av_df.index.min()
    if max_date is None:
        max_date = av_df.index.max()

    av_df = av_df.loc[min_date:max_date]
    fa_df = fa_df.loc[min_date:max_date]
    
    df = av_df.merge(fa_df,how='left',left_on=None, right_on=None,left_index=True, right_index=True)
    
    df = df.reset_index()
    print('Availability and fault datasets merged')
    return(df)

def run_model(df, modeltype = 'RF',random_state = None, num_trees=100, criterion = 'mse',max_depth=None, dtree=True,select='mean',cv=False, visualise=False):
    
    '''function that runs ML models based off chosen features and selected target variable:
    
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
    
    #import general libraries
    
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    #define target variable and features
    
    X = df.drop(['Datetime','Downtime'],axis=1)
    y = df['Downtime']
    
    #train_test_split
   
    from sklearn.model_selection import train_test_split
    
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

            #Tree visulisation

            if visualise == True:

                from IPython.display import Image  
                from six import StringIO  
                from sklearn.tree import export_graphviz
                import pydot 

                features = list(df.columns[2:])

                dot_data = StringIO()  
                export_graphviz(dtree_model, out_file=dot_data,feature_names=features,filled=True,rounded=True)

                graph = pydot.graph_from_dot_data(dot_data.getvalue())  
                Image(graph[0].create_png())  

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

    #Output Test Scatter and Distribution

    plt.scatter(y_test,pred)
    plt.xlabel('Actual Downtime')
    plt.ylabel('Predicted Downtime')
    plt.title('Predicted vs Actual Scatter from Test')

    plt.figure()
    sns.distplot(y_test-pred)
    plt.title('Distrubution of Residuals from Test')
    plt.xlabel('Residual')

    #Output Train Scatter and Distribution

    plt.figure()
    plt.scatter(y_train,model.predict(X_train))
    plt.xlabel('Actual Downtime')
    plt.ylabel('Predicted Downtime')
    plt.title('Predicted vs Actual Scatter from Train')

    plt.figure()
    sns.distplot(y_train-model.predict(X_train))
    plt.title('Distrubution of Residuals from Train')
    plt.xlabel('Residual')


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

        from sklearn.feature_selection import SelectFromModel

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
    
    print('\nRegression Metrics: \n \n', fit_metrics)
    
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

print("Functions Loaded!")