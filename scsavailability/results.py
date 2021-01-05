import pandas as pd
from datetime import datetime

def create_output(fa_PTT,Coeff, speed = 470, picker_present = 0.91, availability = 0.71):

    Output = pd.DataFrame(columns = ['Alert ID','Alert','Fault ID','Asset Code','Tote Colour','Quadrant','MODULE','Original_timestamp'])
    for x in fa_PTT.items():
        df = x[1].merge(Coeff,how = "inner",on="Asset Code")
        df[str(x[0])] = (abs(df['Coefficient']) * df['Duration']) * (speed * picker_present * availability)
        df.drop(['Number','timestamp','PLC','Desk','Duration','Loop','Suffix','PLCN','Alert Type','Pick Station','Coefficient'],axis=1,inplace=True)
        Output = Output.merge(df,how='outer',on=['Alert ID','Alert','Fault ID','Asset Code','Tote Colour','Quadrant','MODULE','Original_timestamp'])
    Output.fillna(0,inplace=True)

    return Output



def create_excel(config, fit_metrics, Coeff, cv_R2):
    """
    Summary
    -------
    Creates table output of model r2 value with metadata and timestamp
    Parameters
    ----------
    model: ModelClass
        fitted model
    start_date: String
        String form of date to filter by in form YYYY-MM-DD
    Returns
    -------
    df_nielsen_clean: pandas DataFrame
        The cleaned and transformed Nielsen data.
    Example
    --------
    df_nielsen_clean = process_model_data(df_nielsen)
    """
    today = datetime.now()
    d1 = today.strftime("%Y%m%d%H%M%S")
    filename = "{0}_{1}.xlsx".format(config.path.save, d1)

    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    df1 = pd.DataFrame(config.toDict())
    df1.to_excel(writer, sheet_name='Sheet1', index=False)
    fit_metrics.to_excel(writer, sheet_name='Sheet2', index=False)
    Coeff.to_excel(writer, sheet_name='Sheet3', index=False)
    cv_R2.to_excel(writer, sheet_name='Sheet4', index=False)

    writer.save()
    print('Saved results: ' + filename)
    return None
    