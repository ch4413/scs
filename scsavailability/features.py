"""
All features work
"""

import pandas as pd
import numpy as np

def create_totes_features(totes_data):

    return None


def create_scs_features(scs_data):

    return None


def pre_process_AT(active_totes):
    
    active_totes = active_totes[~active_totes['MODULE_ASSIGNED'].isin(['ECB', 'RCB'])].copy()
    
    active_totes['MODULE_ASSIGNED'] = active_totes['MODULE_ASSIGNED'].apply(lambda x: x[3:])
    
    active_totes['MODULE_ASSIGNED'] = active_totes['MODULE_ASSIGNED'].apply(pd.to_numeric)
    
    active_totes['DAY'] = active_totes['DAY'].astype('str').str.pad(width=2, side='left', fillchar='0')
    active_totes['HOUR'] = active_totes['HOUR'].astype('str').str.pad(width=2, side='left', fillchar='0')
    active_totes['MINUTE'] = active_totes['MINUTE'].astype('str').str.pad(width=2, side='left', fillchar='0')
    active_totes['timestamp'] = pd.to_datetime(active_totes.apply(
    lambda x: '{0}/{1}/{2} {3}:{4}'.format(x['MONTH'],x['DAY'], x['YEAR'], x['HOUR'], x['MINUTE']), axis=1),dayfirst=True)
    
    active_totes = active_totes.drop(['DAY','MONTH','YEAR','HOUR','MINUTE','ID'],axis=1)
    active_totes.rename(columns = {'MODULE_ASSIGNED':'Module'},inplace = True) 
    
    #Active Totes to Quadrants
    Quad_1 = [int(i) for i in range(1,6)]
    Quad_2 = [int(i) for i in range(7,11)]
    Quad_3 = [int(i) for i in range(11,16)]
    Quad_4 = [int(i) for i in range(17,21)]

    #Assign faults to Quadrants  
    Quad = []

    for i in active_totes['Module']:
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
            
    active_totes['Quadrant']=Quad
    
    print('Active Totes Preprocessed')
        
    return active_totes


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
    
    print('Quadrants Assigned')

    #Assign availability to Modules  
    
    av['Module'] = av['Pick Station'].str[3].astype(int)*10 + av['Pick Station'].str[4].astype(int)
    
    print('Modules Assigned')
    
    av['timestamp'] = pd.to_datetime(av['timestamp'],dayfirst=True)
    
    print("Availability data pre-processed")
    return(av)
    
def preprocess_faults(fa,remove_same_location_faults = True):
    
    fa.columns = pd.Series(fa.columns).str.strip()
    fa.rename(columns = {fa.columns[2]:'timestamp'},inplace = True)

    #Assign PLC code to Quadrants
    Quad_1 = ['C0' + str(i) for i in range(5,8)]  + ['SCSM0' + str(i) for i in range(1,6)]
    Quad_2 = ['C0' + str(i) for i in range(8,10)] + ['SCSM0' + str(i) for i in range(7,10)] + ['SCSM11']
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
    
    #Assign faults to Quadrants  
    Module = []

    for i in range(len(fa)):
        
        if fa['PLC'][i] in ['C05','SCSM01']:
            if fa['Desk'][i] != 'Z':
                Module.append('1')
            else:
                Module.append('C05 External')
                
        elif fa['PLC'][i] in ['C06','SCSM02','SCSM03']:
            if fa['Desk'][i] in ['P01','P02'] or fa['PLC'][i] == 'SCSM02':
                Module.append('2')
            elif fa['Desk'][i] in ['P03','P04'] or fa['PLC'][i] == 'SCSM03':
                Module.append('3') 
            else:
                Module.append('C06 External')
                
        elif fa['PLC'][i] in ['C07','SCSM04','SCSM05']:
            if fa['Desk'][i] in ['P01','P02'] or fa['PLC'][i] == 'SCSM04':
                Module.append('4')
            elif fa['Desk'][i] in ['P03','P04'] or fa['PLC'][i] == 'SCSM05':
                Module.append('5') 
            else:
                Module.append('C07 External')
                
        elif fa['PLC'][i] in ['C08','SCSM07','SCSM08']:
            if fa['Desk'][i] in ['P01','P02'] or fa['PLC'][i] == 'SCSM07':
                Module.append('7')
            elif fa['Desk'][i] in ['P03','P04'] or fa['PLC'][i] == 'SCSM08':
                Module.append('8') 
            else:
                Module.append('C08 External')
                
        elif fa['PLC'][i] in ['C09','SCSM09','SCSM10']:
            if fa['Desk'][i] in ['P01','P02'] or fa['PLC'][i] == 'SCSM09':
                Module.append('9')
            elif fa['Desk'][i] in ['P03','P04'] or fa['PLC'][i] == 'SCSM10':
                Module.append('10') 
            else:
                Module.append('C09 External')
                
        elif fa['PLC'][i] in ['C10','SCSM11']:
            if fa['Desk'][i] != 'Z':
                Module.append('11')
            else:
                Module.append('C10 External')
                
        elif fa['PLC'][i] in ['C11','SCSM12','SCSM13']:
            if fa['Desk'][i] in ['P01','P02'] or fa['PLC'][i] == 'SCSM12':
                Module.append('12')
            elif fa['Desk'][i] in ['P03','P04'] or fa['PLC'][i] == 'SCSM13':
                Module.append('13') 
            else:
                Module.append('C11 External')
                
        elif fa['PLC'][i] in ['C12','SCSM14','SCSM15']:
            if fa['Desk'][i] in ['P01','P02'] or fa['PLC'][i] == 'SCSM14':
                Module.append('14')
            elif fa['Desk'][i] in ['P03','P04'] or fa['PLC'][i] == 'SCSM15':
                Module.append('15') 
            else:
                Module.append('C12 External')
                
        elif fa['PLC'][i] in ['C13','SCSM17','SCSM18']:
            if fa['Desk'][i] in ['P01','P02'] or fa['PLC'][i] == 'SCSM17':
                Module.append('17')
            elif fa['Desk'][i] in ['P03','P04'] or fa['PLC'][i] == 'SCSM18':
                Module.append('18') 
            else:
                Module.append('C13 External')
                
        elif fa['PLC'][i] in ['C14','SCSM19','SCSM20']:
            if fa['Desk'][i] in ['P01','P02'] or fa['PLC'][i] == 'SCSM19':
                Module.append('19')
            elif fa['Desk'][i] in ['P03','P04'] or fa['PLC'][i] == 'SCSM20':
                Module.append('20') 
            else:
                Module.append('C14 External')       
           
                
        elif fa['PLC'][i] in ['C' + str(i) for i in range(35,54)]:       
            Module.append('Destacker')
                
        elif fa['PLC'][i] in ['C17','SCSM22']:
            Module.append('ECB')     
                
        elif fa['PLC'][i] in ['C15','C16','C23']:
            Module.append('Outer Loop')
        else:
            Module.append('PLC code not from SCS')
                
    fa['Module']=Module
    
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
        
    print('Faults Preprocessed')
    
    return(fa)


def floor_shift_time_fa(df,time_col = 'timestamp',floor_units = 'H',shift=0, shift_units='m'):
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
    
    df[time_col] = df[time_col].apply(lambda x:x+pd.to_timedelta(shift,unit=shift_units))
    
    print('Time shifted by ' + str(shift) + shift_units)
    
    #floors units to round down to the nearest specified time interval (Hour by default)
    
    df[time_col] = df[time_col].dt.floor(floor_units)
    return(df)


def fault_select(fa, select_level, selection):
    
    fa = fa.copy()
    fa = fa[fa[select_level].isin(selection)] 
    
    return fa
    
    
def faults_aggregate(df,fault_agg_level , agg_col = 'Duration',agg_type = 'count' ,time_col = 'timestamp',break_durations = False):
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
        if fault_agg_level == None:
            
            df = df.groupby(time_col,as_index = False).agg({agg_col:agg_type})
            df.rename(columns = {'Duration':'Total Faults'},inplace = True)
            df = df.set_index('timestamp')
            
        if fault_agg_level != None:
    
            df = df.groupby([time_col,fault_agg_level],as_index = False).agg({agg_col:agg_type})
            df = pd.pivot_table(df,values = agg_col,index = time_col,columns = fault_agg_level,fill_value=0)
   
    print('Faults aggregated')
    return(df)

def av_at_select(av, at, select_level =None, selection = None, remove_high_AT = False):

    av = av.copy()
    at = at.copy()
    
    if select_level != None:
        print("Availability and Totes will contain data only for " + str(select_level) + ": " + str(selection))
        
        if select_level == 'Pick Station':
            
            av = av[av['Pick Station']==selection]
        
        
        elif select_level == 'Module' or select_level == 'Quadrant':
                
            av = av[av[select_level]==selection]
            at = at[at[select_level]==selection]
            
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
                
    return av,at            


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
   
    if agg_level == None:
    
        df = df.groupby(['timestamp'],as_index=False).agg({'Downtime':'mean','Blue Tote Loss':'mean','Grey Tote Loss':'mean'})
    else:
        df = df.groupby(['timestamp',agg_level],as_index=False).agg({'Downtime':'mean','Blue Tote Loss':'mean','Grey Tote Loss':'mean'})
        
    df = df.set_index('timestamp')
    print('Availability data aggregated')
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

def merge_av_fa_at(av_df,fa_df=None,at_df=None,min_date=None,max_date=None, target = 'Downtime',faults=True, totes = True,agg_level=None):
    '''
    function that merges availability and fault datasets by date index
    '''
    
    av_df = av_df.copy()
    fa_df = fa_df.copy()
    at_df = at_df.copy()
    
    
    if min_date is None:
        min_date = av_df.index.min()
    if max_date is None:
        max_date = av_df.index.max()
        
        
    if agg_level == None:
    
        av_df = pd.DataFrame(av_df[~av_df[target].isnull()][target]).loc[min_date:max_date]

        if faults == True and totes == False:

            df = av_df.merge(fa_df,how='inner',left_on=None, right_on=None,left_index=True, right_index=True)

        elif faults == False and totes == True:

            df = av_df.merge(at_df,how='inner',left_on=None, right_on=None,left_index=True, right_index=True)

        else:

            df = av_df.merge(fa_df,how='inner',left_on=None, right_on=None,left_index=True, right_index=True)
            df = df.merge(at_df,how='inner',left_on=None, right_on=None,left_index=True, right_index=True)

        df.reset_index(inplace=True)

    if agg_level != None:
        
        av_df = pd.DataFrame(av_df[~av_df[target].isnull()][[target,agg_level]]).loc[min_date:max_date]
        
        av_df.reset_index(inplace=True)
        at_df.reset_index(inplace=True)
        fa_df.reset_index(inplace=True)
        
        if faults == True and totes == False:
            
            df = av_df.merge(fa_df,how='inner',on = 'timestamp')

        elif faults == False and totes == True:

            df = av_df.merge(at_df,how='inner', on = ['timestamp',agg_level])

        else:
            df = av_df.merge(fa_df,how='inner',on = 'timestamp')
            df = df.merge(at_df,how='inner', on = ['timestamp',agg_level])

         
        df.drop([agg_level],axis=1,inplace=True)

    #remove columns with only zeros (faults that did not happen in this period of time or quadrant)
    df = df.loc[:, (df != 0).any(axis=0)]
    print('Datasets merged')
    return(df)