
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
    (e.g., if you only want observations from quadrant 1)'''
    
    
    if break_durations == True:
        df['Duration_category'] = df.groupby("fault ID type")["Duration"].apply(lambda x: pd.cut(x, 3,labels=['short','medium','long']))
        cols = [fault_level,'Duration_category']
        df['period'] = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        df = df.groupby([time_col,'Quadrant','period'],as_index = False).agg({agg_col:agg_type})
        if quadrant != None:
            df = df[df['Quadrant'].isin([quadrant, 0])]
        df = pd.pivot_table(df,values = agg_col,index = time_col,columns = ['period'],fill_value=0)
        print('Duration broken into categories Short, Medium, Long')
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

def run_model(df,num_trees=100,dtree=True,select=True,visualise=False):
    
    '''function that runs ML models based off chosen features and selected target variable:
    
    1. performs test train split,
    2. runs decision tree model (if required), 
    3. outputs decision tree visulation (if required),
    4. runs random forest model
    5. outputs residual distrubution, scatter plot and feature importance
    6. runs selected random forest model (if required)
    7. outputs regression metrics from model(s) 
    
    Parameters:
    
    df: input data frame containing features and target variable
    num_trees: number of tree for random forrest model (100 by default)
    dtree: option for running decision tree model (True by default)
    select: option for running selected random forest model (True by default)
    visualise: option for outputing decision tree visualisation (False by default)
    
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    #Set up metrics dataframe
    
    fit_metrics = pd.DataFrame(index = ['MAE','MSE','RMSE','MAPE%','ACC%','OOB','R2_Train','R2_Pred'])
    
    #import metrics
    
    from sklearn import metrics
    
    if dtree==True:
    
        #Fit decision Tree

        from sklearn.tree import DecisionTreeRegressor

        dtree = DecisionTreeRegressor()
        dtree.fit(X_train,y_train)

        #Predicting using decision tree

        dtree_pred = dtree.predict(X_test)

        #Fill dataframe with metrics

        mape = np.mean(np.abs((y_test - dtree_pred) / np.abs(y_test)))

        fit_metrics['D_Tree'] = [metrics.mean_absolute_error(y_test, dtree_pred),metrics.mean_squared_error(y_test, dtree_pred),
                                np.sqrt(metrics.mean_squared_error(y_test, dtree_pred)),round(mape * 100, 2),round(100*(1 - mape), 2),
                                'N/A',dtree.score(X_train,y_train),dtree.score(X_test,y_test)]
    
    #Tree visulisation
    
    if visualise == True:
    
        from IPython.display import Image  
        from six import StringIO  
        from sklearn.tree import export_graphviz
        import pydot 

        features = list(df.columns[2:])
    
        dot_data = StringIO()  
        export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

        graph = pydot.graph_from_dot_data(dot_data.getvalue())  
        Image(graph[0].create_png())  
    
    #Fit Random Forest

    from sklearn.ensemble import RandomForestRegressor

    rfr = RandomForestRegressor(n_estimators=num_trees,oob_score = True)
    rfr.fit(X_train, y_train)
    
    #Predicting using random forest
    
    rf_pred = rfr.predict(X_test)
    
    #Fill dataframe with metrics
    
    mape = np.mean(np.abs((y_test - rf_pred) / np.abs(y_test)))
    
    fit_metrics['RF'] = [metrics.mean_absolute_error(y_test, rf_pred),metrics.mean_squared_error(y_test, rf_pred),
                            np.sqrt(metrics.mean_squared_error(y_test, rf_pred)),round(mape * 100, 2),round(100*(1 - mape), 2),
                            rfr.oob_score_,rfr.score(X_train,y_train),rfr.score(X_test,y_test)]
    
    #Output Scatter and Distribution
    
    plt.scatter(y_test,rf_pred)
    plt.xlabel('Actual Downtime')
    plt.ylabel('Predicted Downtime')
    plt.title('Predicted vs Actual Scatter from RF')
    
    plt.figure()
    sns.distplot(y_test-rf_pred)
    plt.title('Distrubution of Residuals from RF')
    plt.xlabel('Residual')
    
    #Output Feature Importance
    
    Importance = pd.DataFrame({'Importance': rfr.feature_importances_}, index=X.columns).sort_values(by='Importance', ascending=False)
    plt.figure()
    sns.barplot(data = Importance.head(10), x= Importance.head(10).index, y='Importance')
    plt.xlabel('Feature')
    print(Importance.head(10))
    
    if select == True:
    
        #Reducing Demensionality

        #Fit select model

        from sklearn.feature_selection import SelectFromModel

        sel = SelectFromModel(RandomForestRegressor(n_estimators = num_trees))
        sel.fit(X_train, y_train)

        #Set selected features

        selected_feat= X_train.columns[(sel.get_support())]

        #Reduce number of features

        X_sel = df[selected_feat]

        #Train Test Split

        X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3)

        #Fit reduced model

        rfr_sel = RandomForestRegressor(n_estimators=num_trees,oob_score=True)
        rfr_sel.fit(X_train,y_train)

        #Predicting using random forest

        rf_sel_pred = rfr_sel.predict(X_test)

        #Fill dataframe with metrics

        mape = np.mean(np.abs((y_test - rf_sel_pred) / np.abs(y_test)))

        fit_metrics['RF Reduced'] = [metrics.mean_absolute_error(y_test, rf_sel_pred),metrics.mean_squared_error(y_test, rf_sel_pred),
                                np.sqrt(metrics.mean_squared_error(y_test, rf_sel_pred)),round(mape * 100, 2),round(100*(1 - mape), 2),
                                rfr_sel.oob_score_,rfr_sel.score(X_train,y_train),rfr_sel.score(X_test,y_test)]
    
    #print metrics
    
    print('\n', fit_metrics)

print("Functions Loaded!")