import pandas as pd
from datetime import datetime
import numpy as np
import scsavailability as scs
from . import logger
    
from scsavailability import features as feat, model as md, results as rs

@logger.logger
def create_output(fa_PTT,Coeff, report_start, report_end, speed = 470, picker_present = 0.91, availability = 0.71):

    Output = pd.DataFrame(columns = ['Alert ID','Alert','Fault ID','Asset Code','Tote Colour','Quadrant','MODULE','Entry Time'])
    for x in fa_PTT.items():
        df = x[1].merge(Coeff,how = "inner",on="Asset Code")
        df['Downtime'] = abs(df['Coefficient']) * df['Duration']
        df['ones'] = pd.Series(np.ones(len(df)))
        df['PTT'] = str(x[0])
        df['Singles'] = df[['Downtime','ones']].min(axis=1) * (speed * picker_present * availability)
        df.drop(['timestamp','Duration','Loop','Suffix','PLCN','Alert Type','Pick Station','Coefficient','Downtime','ones'],axis=1,inplace=True)
        Output = pd.concat([Output,df],join='outer',ignore_index=True)
    Output.fillna(0,inplace=True)

    Output = Output[(Output['Entry Time']>=report_start) & (Output['Entry Time']<report_end)]

    Final_Output = pd.DataFrame({'NUMBER': Output['Number'], 
                                'ALERT':Output['Alert'],
                                'ENTRY_TIME':Output['Entry Time'],
                                'END_TIME':Output['End Time'],
                                'PLC':Output['PLC'],
                                'DESK':Output['Desk'],
                                'FAULT_ID':Output['Fault ID'],
                                'ID':Output['Asset Code'],
                                'AREA':Output['Area'],
                                'BLUEGREY':Output['Tote Colour'],
                                'PTT': Output['PTT'],
                                'SINGLES': Output['Singles']
                                })

    return Final_Output

@logger.logger
def run_single_model(at,av,fa,report_start,report_end,shift,weights,speed,picker_present,availability,duration_thres=0):

    fa_floor = feat.floor_shift_time_fa(fa, shift=shift,duration_thres=duration_thres)

    df,fa_PTT = feat.create_PTT_df(fa_floor,at,av,weights=weights)
    df = feat.log_totes(df)

    X,y = md.gen_feat_var(df,target = 'Availability', features = ['Totes','Faults'])
    X_train, X_test, y_train, y_test = md.split(X,y,split_options = {'test_size': 0.3,
                                                                    'random_state': None})

    R2_OOS,Coeff, Num_Assets = md.run_OLS(X_train = X_train,y_train = y_train,X_test = X_test,y_test=y_test, n = 30)

    Output = rs.create_output(fa_PTT,Coeff,report_start,report_end,speed = speed, picker_present = picker_present, availability = availability)

    return Output, R2_OOS, Num_Assets