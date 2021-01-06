import pandas as pd
from datetime import datetime
import numpy as np

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
