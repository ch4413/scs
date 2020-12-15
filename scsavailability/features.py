"""
All features work
"""

import pandas as pd
import numpy as np
from . import logger
import pkg_resources

def create_totes_features(totes_data):

    return None


def create_scs_features(scs_data):

    return None

@logger.logger
def load_module_lookup():
    """Return a dataframe about the 68 different Roman Emperors.

    Contains the following fields:
        index          68 non-null int64
        name           68 non-null object
        name.full      68 non-null object
    ... (docstring truncated) ...

    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream(__name__, 'data/module_lookup.csv')
    return pd.read_csv(stream)

@logger.logger
def load_tote_lookup():
    """Return a dataframe about the 68 different Roman Emperors.

    Contains the following fields:
        index          68 non-null int64
        name           68 non-null object
        name.full      68 non-null object
    ... (docstring truncated) ...

    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream(__name__, 'data/tote_lookup.csv')
    return pd.read_csv(stream)

@logger.logger
def pre_process_AT(active_totes):
    
    active_totes = active_totes[~active_totes['MODULE_ASSIGNED'].isin(['ECB', 'RCB'])].copy()
    
    active_totes['MODULE_ASSIGNED'] = active_totes['MODULE_ASSIGNED'].apply(lambda x: x[3:])
    
    active_totes['MODULE_ASSIGNED'] = active_totes['MODULE_ASSIGNED'].apply(pd.to_numeric)
    
    active_totes['DAY'] = active_totes['DAY'].astype('str').str.pad(width=2, side='left', fillchar='0')
    active_totes['HOUR'] = active_totes['HOUR'].astype('str').str.pad(width=2, side='left', fillchar='0')
    active_totes['MINUTE'] = active_totes['MINUTE'].astype('str').str.pad(width=2, side='left', fillchar='0')
    active_totes['timestamp'] = pd.to_datetime(active_totes.apply(
    lambda x: '{0}/{1}/{2} {3}:{4}'.format(x['DAY'],x['MONTH'], x['YEAR'], x['HOUR'], x['MINUTE']), axis=1),dayfirst=True)
    
    active_totes = active_totes.drop(['DAY','MONTH','YEAR','HOUR','MINUTE','ID'],axis=1)
    active_totes.rename(columns = {'MODULE_ASSIGNED':'Module'},inplace = True) 
    
    #Active Totes to Quadrants
    active_totes['Quadrant'] = 0
    active_totes.loc[active_totes['Module'] < 6, 'Quadrant'] = 1
    active_totes.loc[(active_totes['Module'] > 6)
        & (active_totes['Module'] < 11), 'Quadrant'] = 2
    active_totes.loc[(active_totes['Module'] > 10)
        & (active_totes['Module'] < 16), 'Quadrant'] = 3
    active_totes.loc[(active_totes['Module'] > 16)
        & (active_totes['Module'] < 21), 'Quadrant'] = 4
        
    return active_totes

@logger.logger
def pre_process_av(av):
    '''
    function that pre-processes the raw csv files:
    1. converts date columns to datetime objects,
    2. assigns quandrants and modules, 
    3. drops rows with missing values, 
    4. drops faults that happen in same location at same time (keeps fault with max duration) - if selected 
    
    Note: function will need to be adapated to preprocess availability data from other months
    '''
    
    #convert to percentage downtime
    
    av['Availability'] = 1 - av['Availability']
    av.rename(columns = {'Availability':'Downtime',av.columns[0]:'timestamp'},inplace = True)

    #Assign Pick Station to Quadrant
    Quad_1 = ['PTT011','PTT012','PTT021','PTT022','PTT031','PTT032','PTT041','PTT042','PTT051','PTT052']
    Quad_2 = ['PTT071','PTT072','PTT081','PTT082','PTT091','PTT092','PTT101','PTT102']
    Quad_3 = ['PTT111','PTT112','PTT121','PTT122','PTT131','PTT132','PTT141','PTT142','PTT151','PTT152']
    Quad_4 = ['PTT171','PTT172','PTT181','PTT182','PTT191','PTT192','PTT201','PTT202']


    #Assign availability to Quadrants  
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
    av['Quadrant'] = Quad
    
    #print('Quadrants Assigned')

    #Assign availability to Modules  
    
    av['Module'] = av['Pick Station'].str[3].astype(int)*10 + av['Pick Station'].str[4].astype(int)
    
    #print('Modules Assigned')
    
    av['timestamp'] = pd.to_datetime(av['timestamp'],dayfirst=True)
    
    #print("Availability data pre-processed")
    return(av)

@logger.logger
def preprocess_faults(fa,remove_same_location_faults = True):
    
    fa.columns = pd.Series(fa.columns).str.strip()
    fa.rename(columns = {fa.columns[2]:'timestamp'},inplace = True)

    #Assign PLC code to Quadrants
    Quad_1 = ['C0' + str(i) for i in range(5,8)]  + ['SCSM0' + str(i) for i in range(1,6)]
    Quad_2 = ['C0' + str(i) for i in range(8,10)] + ['SCSM0' + str(i) for i in range(7,10)] + ['SCSM10']
    Quad_3 = ['C'  + str(i) for i in range(10,13)] + ['SCSM' + str(i) for i in range(11,16)]
    Quad_4 = ['C'  + str(i) for i in range(13,15)] + ['SCSM' + str(i) for i in range(17,21)]


    #Assign faults to Quadrants  
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

    lu = load_module_lookup()
    # Copy desk
    fa['Desk_edit'] = fa['Desk']
    # Mark SCSs
    fa.loc[fa['PLC'].str.contains(r'SCS', regex=True), 'Desk_edit'] = fa.loc[fa['PLC'].str.contains(r'SCS', regex=True), 'PLC']
    # fa.loc[fa['PLC'].str.contains(r'SCS(?!\ M)', regex=True), 'Desk_edit'] = fa.loc[fa['PLC'].str.contains(r'SCS(?!\ M)', regex=True), 'Desk']
    # fa.loc[fa['PLC'].str.contains(r'SCS\ M', regex=True), 'Desk_edit'] = fa.loc[fa['PLC'].str.contains(r'SCS\ M', regex=True), 'PLC']
    # Mark PTTs
    fa.loc[~(fa['Pick Station']==False), 'Desk_edit'] = fa[~(fa['Pick Station']==False)]['Pick Station'].apply(lambda x: x[:-1])
    # Set NA desk for outside stuff
    fa.loc[fa['PLC'].isin(['C23', 'C16', 'C15', 'C17']), 'Desk_edit'] = 'X'
    #fa['PLCN'] = fa['PLC'].str.extract('((?<=C)[0-9]{2})')[0].astype('float')
    fa.loc[fa['PLCN'] > 34, 'Desk_edit'] = 'X'
    fa = pd.merge(fa, lu, how='left', on=['PLC', 'Desk_edit']).drop('Desk_edit', axis=1)
    fa['timestamp'] = pd.to_datetime(fa['timestamp'],dayfirst=True)
    
    #drop rows where there is no duration data
    fa = fa.dropna(subset = ['Duration'])

    #convert duration string to time delta and then to seconds (float)
    fa['Duration'] = pd.to_timedelta(fa['Duration'].str.slice(start=2))
    fa['Duration'] = fa['Duration'].dt.total_seconds()
    
    #drop faults that happen at same time and in same location (keep only the one with max duration)
    if remove_same_location_faults == True:
        fa = fa.sort_values('Duration').drop_duplicates(subset=['timestamp', 'PLC', 'Desk'],keep='last')
        print('duplicated location faults removed - max duration kept')
    
    ## !!HOTFIX
    fa.loc[fa['Loop']=='Quadrant', 'MODULE'] = np.nan
    print('HOTFIX: Quadrant only faults')
    return fa

def floor_shift_time_fa(df,shift=0):
    '''
    function that shifts and floors datetime column in a pandas df to the specific time unit
    Note: defaults to HOUR
    
    Parameters:
    
    df: input dataframe
    time_col: column name with fault entry time
    floor_units: units to floor on (Default Hour)
    shift: units to shift time by (No shift by default)
    shift_units: units to shift by (minutes by default)
    
    '''
    
    df = df.copy()
    
    #Shifts entry time by desired amount
    
    df['timestamp'] = df['timestamp'].apply(lambda x:x+pd.to_timedelta(shift,unit='m'))
    
    print('Time shifted by ' + str(shift) + 'Minutes')
    
    #floors units to round down to the nearest specified time interval (Hour by default)
    
    df['timestamp'] = df['timestamp'].dt.floor('H')

    return df 


def fault_select(fa, fault_select_options=None,duration_thres = 0):
    
    fa = fa.copy()
    
    if fault_select_options != None:
       
        for i in fault_select_options.keys():
    
            fa = fa[fa[i].isin(fault_select_options[i])] 

    fa = fa[fa['Duration']>duration_thres]
    
    return fa
    
def faults_aggregate(df, fault_agg_level, agg_type = 'count'):
    '''
    function that aggregates fault data by specified metric (agg_type) and quadrant.
    - The quadrant parameter is used in case you want to filter for a specifc quadrant
    (e.g., if you only want observations from quadrant 1)
    - agg type is count/mean/sum
    -fault_level is 'fault ID type' or 'Number' or 'PLC' or 'Quadrant'
    -agg_col should be duration
    -break_durations if you want to bucket up durations in short, medium and long
    '''
    
    df = df.copy()
  
    if fault_agg_level == 'None':

        df = df.groupby('timestamp',as_index = False).agg({'Duration':agg_type})
        df.rename(columns = {'Duration':'Total Faults'},inplace = True)
        df = df.set_index('timestamp')

    else:

        df = df.groupby(['timestamp',fault_agg_level],as_index = False).agg({'Duration':agg_type})
        df = pd.pivot_table(df,values = 'Duration',index = 'timestamp',columns = fault_agg_level,fill_value=0)
   
    return df 

def av_at_select(av, at, availability_select_options = None,remove_high_AT = True, AT_limit = None):

    av = av.copy()
    at = at.copy()
    
    if availability_select_options != None:
       
        for i in availability_select_options.keys():
    
            if i == 'Pick Station':

                av = av[av[i].isin(availability_select_options[i])]


            elif i == 'Module' or i == 'Quadrant':

                av = av[av[i].isin(availability_select_options[i])]
                at = at[at[i].isin(availability_select_options[i])]

            else:

                print('\nNot a valid level, returned all data\n')          
            
    if remove_high_AT == True:
        
        at_piv = pd.pivot_table(at,values = 'TOTES',index = 'timestamp', columns = 'Module')
        at_lookup = at.groupby('Module').mean().drop('Quadrant',axis=1)
        
        Limit = []

        for i in at_piv.columns: 

            Q1 = at_piv[i].quantile(0.25)
            Q3 = at_piv[i].quantile(0.75)
            Limit.append(Q3 + 1.5 * (Q3-Q1))
            
        at_lookup['Upper limit'] = Limit
            
        at_lookup.drop('TOTES',axis=1,inplace=True)
        
        at = at.join(at_lookup,how='inner',on='Module')
        
        at = at[at['TOTES']<=at['Upper limit']]
    
        at.drop('Upper limit',axis=1,inplace=True)
        
    if AT_limit != None:
        at.reset_index(inplace=True)
        for i in range(len(at['TOTES'])):
            if at['TOTES'][i]>AT_limit:
                at['TOTES'][i]=AT_limit
                
    return av, at            

@logger.logger
def aggregate_availability(df, agg_level = None):
    '''
    function to aggregate availability at chosen level:
    
    1. Selects availability data revelevent to chosen level
    2. Aggregates Availability Data
    
    Parameters:
    
    df: input dataframe
    time_col: column name containing time
    level: which level to aggregate at (i.e. Quadrant/Module/Pick Station)
    selected: selected area within that level (i.e. Quadrant 1/Module 1/PPT011)
    
    '''
    df = df.copy()
   
    if agg_level == 'None':
    
        df = df.groupby(['timestamp'],as_index=False).agg({'Downtime':'mean','Blue Tote Loss':'mean','Grey Tote Loss':'mean'})
    else:
        df = df.groupby(['timestamp',agg_level],as_index=False).agg({'Downtime':'mean','Blue Tote Loss':'mean','Grey Tote Loss':'mean'})
        
    df = df.set_index('timestamp')
#    print('Availability data aggregated')
    return(df)

def aggregate_totes(active_totes, agg_level = None):
    
    active_totes = active_totes.copy()
    
    active_totes['timestamp'] = active_totes['timestamp'].dt.floor('H')
    
    if agg_level == 'Module':
    
        active_totes = active_totes.groupby(['timestamp','Module'],as_index=False).mean()
        active_totes.drop('Quadrant',axis=1,inplace=True)
        
    elif agg_level == 'Quadrant':
        
        active_totes = active_totes.groupby(['timestamp','Quadrant'],as_index=False).mean()
        active_totes.drop('Module',axis=1,inplace=True)
    
    else:
        active_totes = active_totes.groupby('timestamp',as_index=False).mean()
        active_totes.drop(['Module','Quadrant'],axis=1,inplace=True)
        
        
    active_totes = active_totes.set_index('timestamp')    
        
    return(active_totes)


def weight_hours(df, weights = [1,0.5,0.25]):
    '''
    function to include weighted fault data from previous hours
    
    Parameters:
    
    df: input data frame
    weights: weights for hours with first element in array being weight for current hour, second the previous hour etc.
    '''
    
    df_weight = pd.DataFrame(data = np.zeros(df.shape),index=df.index,columns = df.columns)
    
    for i in range(len(df)):
        
        #iterate through each row in orginal df required in row of new df
        
        for x in range(len(weights)):
            
            if i-x >= 0:
         
                df_weight.iloc[i] = df_weight.iloc[i] + df.iloc[i-x]*weights[x]

    print('Previous Hours Weighted')
    return(df_weight)

def merge_av_fa_at(av_df,fa_df,at_df,min_date=None,max_date=None,agg_level=None):
    '''
    function that merges availability and fault datasets by date index
    '''
    
    av_df = av_df.copy()
    fa_df = fa_df.copy()
    at_df = at_df.copy()
    
    if min_date != None:
    
        min_date = max(av_df.index.min(),fa_df.index.min(),at_df.index.min(),min_date)
        
    else:
        
        min_date = max(av_df.index.min(),fa_df.index.min(),at_df.index.min())
        
    if max_date != None:    

        max_date = min(av_df.index.max(),fa_df.index.max(),at_df.index.max(),max_date)
        
    else:
        

        max_date = min(av_df.index.max(),fa_df.index.max(),at_df.index.max())
    
    fa_df = fa_df.loc[min_date:max_date]

    if agg_level == 'None':

    
        av_df = av_df[["Downtime","Blue Tote Loss","Grey Tote Loss"]].loc[min_date:max_date]
        

        df = av_df.merge(fa_df,how='inner',left_on=None, right_on=None,left_index=True, right_index=True)

        df = df.merge(at_df,how='inner',left_on=None, right_on=None,left_index=True, right_index=True)
        
        df.reset_index(inplace=True)

    if agg_level != 'None':
        
        av_df = av_df[["Downtime","Blue Tote Loss","Grey Tote Loss", agg_level]].loc[min_date:max_date]
        
        av_df.reset_index(inplace=True)
        at_df.reset_index(inplace=True)
        fa_df.reset_index(inplace=True)
        
        
        df = av_df.merge(fa_df,how='inner',on = 'timestamp')
        df = df.merge(at_df,how='inner', on = ['timestamp',agg_level])
        
        df.drop([agg_level],axis=1,inplace=True)
        
    return df   

def add_code(data):
    """
    Summary
    -------
    Takes variables and fits model with arguments. Return model object.
    Parameters
    ----------
    data: pandas DataFrame
        dataframe of features
    Returns
    -------
    scs: pandas DataFrame
        dataframe with 'code'
    Example
    --------
    scs = add_code(data)
    """
    scs = data.copy()
    scs['Asset Code'] = scs['Alert'].str.extract('(^[A-Z]{3}[0-9]{3}|[A-Z][0-9]{4}[A-Z]{3}[0-9]{3}|[A-Z]{3} [A-Z][0-9]{2})')
    scs['Asset Code'] = scs['Alert'].str.extract('([A-Z][0-9]{4}[A-zZ]{3}[0-9]{3})')
    scs.loc[scs['PLC'].str.contains(r'SCS', regex=True), 'Asset Code'] = scs.loc[scs['PLC'].str.contains(r'SCS', regex=True), 'Desk']
    scs.loc[scs['Asset Code'].isna(), 'Asset Code'] = scs.loc[scs['Asset Code'].isna(), 'PLC']
    
    return scs

def add_tote_colour(scs_code):
    """
    Summary
    -------
    Takes variables and fits model with arguments. Return model object.
    Parameters
    ----------
    scs_code: pandas DataFrame
        dataframe of features
    asset_lu: pandas DataFrame
        dataframe
    Returns
    -------
    scs: pandas DataFrame
        dataframe with 'code'
    Example
    --------
    scs, unmapped = add_tote_colour(data)
    """
    asset_lu = load_tote_lookup()
    df_totes = pd.merge(scs_code, asset_lu.drop('Number', axis=1), how='left', on='Asset Code')
    df_totes.loc[df_totes['PLC'].isin(['C17', 'C16', 'C15', 'C23']), 'Tote Colour'] = 'Blue'
    df_totes['Pick Station'] = df_totes['Alert'].str.extract('(PTT[0-9]{3})').fillna(False)
    df_totes.loc[(df_totes['Pick Station']!=False), 'Tote Colour'] = 'Both'
    df_totes['PLCN'] = df_totes['PLC'].str.extract('((?<=C)[0-9]{2})').fillna(0).astype('int')
    df_totes.loc[df_totes['PLCN'] > 34, 'Tote Colour'] = 'Blue'
    
    # Unmapped
    unmapped = df_totes[df_totes['Tote Colour'].isna()]['Asset Code'].value_counts().reset_index().copy()
    unmapped = unmapped.rename(columns={'index' : 'Asset', 'Asset Code' : 'Occurrence'})
    # Map unknown to Both
    df_totes.loc[df_totes['Tote Colour'].isna(), 'Tote Colour'] = 'Both'

    return df_totes, unmapped

def get_data_faults(data, modules):
    """
    Summary
    -------
    Anything in same Module
    PLC External applying to that PLC
    Quadrant loop dataults for the module that quadrant is in
    Outer

    Parameters
    ----------
    data: pandas DataFrame
        dataframe of features
    module: numeric
        dataframe
    Returns
    -------
    scs: pandas DataFrame
        dataframe with 'code'
    Example
    --------
    scs, unma
    """
    #1
    mod_str = pd.Series(modules).astype('str')
    faults1 = data[data['MODULE'].isin(mod_str)]
    #2
    a = data[['PLC', 'MODULE']].drop_duplicates()
    b = a[(a['MODULE'].isin(mod_str)) & a['PLC'].str.contains('^C', regex=True)]
    c = b['PLC'] + ' External'
    faults2 = data[data['MODULE'].isin(list(c))]
    #3
    ## Module can't be assigned if Loop == Quadrant
    q = data[data['MODULE'].isin(mod_str)]['Quadrant']
    faults3 = data[(data['Loop'].isin(['Quadrant'])) & data['Quadrant'].isin(q)]
    #4
    faults4 = data[data['Loop'].isin(['Outside'])]
    faults_mod = pd.concat([faults1, faults2, faults3, faults4])
    
    return faults_mod