import pandas as pd
from datetime import datetime
import numpy as np
import scsavailability as scs
    
from scsavailability import features as feat, model as md, plotting as pt, results as rs

def create_output(fa_PTT,Coeff, end_time, speed = 470, picker_present = 0.91, availability = 0.71):

    Output = pd.DataFrame(columns = ['Alert ID','Alert','Fault ID','Asset Code','Tote Colour','Quadrant','MODULE','Original_timestamp'])
    for x in fa_PTT.items():
        df = x[1].merge(Coeff,how = "inner",on="Asset Code")
        df['Downtime'] = abs(df['Coefficient']) * df['Duration']
        df['ones'] = pd.Series(np.ones(len(df)))
        df[str(x[0])] = df[['Downtime','ones']].min(axis=1) * (speed * picker_present * availability)
        df.drop(['Number','timestamp','PLC','Desk','Duration','Loop','Suffix','PLCN','Alert Type','Pick Station','Coefficient','Downtime','ones'],axis=1,inplace=True)
        Output = Output.merge(df,how='outer',on=['Alert ID','Alert','Fault ID','Asset Code','Tote Colour','Quadrant','MODULE','Original_timestamp'])
    Output.fillna(0,inplace=True)

    time_limit = end_time - pd.to_timedelta(12, unit='h')

    Output = Output[Output['Original_timestamp']>time_limit]

    return Output


def run_single_model(at,av,fa,end_time,shift,weights,speed,picker_present,availability):

    fa_floor = feat.floor_shift_time_fa(fa, shift=shift)

    df,fa_PTT = feat.create_PTT_df(fa_floor,at,av,weights=weights)
    df = feat.log_totes(df) 
    df_2week = df[df['timestamp']>end_time - pd.to_timedelta(14, unit='D')]

    X,y = md.gen_feat_var(df_2week,target = 'Availability', features = ['Totes','Faults'])
    X_train, X_test, y_train, y_test = md.split(X,y,split_options = {'test_size': 0.3,
                                                                    'random_state': None})

    R2_cv,R2_OOS,Coeff = md.run_OLS(X_train = X_train,y_train = y_train,X_test = X_test,y_test=y_test, n = 30)

    Output = rs.create_output(fa_PTT,Coeff,end_time,speed = speed, picker_present = picker_present, availability = availability)

    return Output, R2_OOS