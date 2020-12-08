import pandas as pd
from datetime import datetime

def create_output(config, fit_metrics, Coeff, cv_R2):
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
    