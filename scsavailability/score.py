'''
Script for scoring model metrics
'''

import numpy as np
import pandas as pd
from scipy import stats

def lm_coefficients(model, X, y):
    """
    Summary
    -------
    Creates table output of model r2 value with metadata and timestamp
    Parameters
    ----------
    model: ModelClass
        fitted model
    X: 
        pandas DataFrame or numpy array
    y: 
    start_date: String
    Returns
    -------
    myDF3: pandas DataFrame
        dataframe with p-value and t-values
    Example
    --------
    mod_results = lm_coefficients(model, X, y)
    """
    lm = model
    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X)

    newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

    # Note if you don't want to use a DataFrame replace the two lines above with
    # newX = np.append(np.ones((len(X),1)), X, axis=1)
    # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b

    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX.loc[0])))) for i in ts_b]

    sd_b = np.round(sd_b,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilities"] = [params,sd_b,ts_b,p_values]
    coeff_names = pd.concat([pd.Series(['Constant']), pd.Series(X.columns)]).reset_index(drop=True)
    myDF3['Feature'] = coeff_names
    myDF3 = myDF3[['Feature', 'Coefficients', 'Standard Errors', 't values', 'Probabilities']]
    
    return myDF3
