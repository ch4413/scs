import pandas as pd
from . import logger
import warnings
from copy import copy
from scsavailability import model as md, results as rs


@logger.logger
def create_output(fa_PTT, coeff):
    """
    Summary
    -------
    Create output dataframe of lost single due
    to faults over reporting window from
    coefficients and pick station faults dictionary
    ----------
    fa_PTT: dictionary
        dictionary of pick station specific fault dataframes
    coeff: pandas DataFrame
        Selected assets and coefficients from ML Model
    Returns
    -------
    output: pandas DataFrame
        dataframe containing labeled selected faults
        over reporting window with the number of losts singles
        due to each fault at pick stations
    Example
    --------
    output = create_output(fa_PTT,
                           coeff)
    """
    # Create empty dataframe to fill
    output = pd.DataFrame(columns=['Asset Code', 'Coefficient', 'PTT'])

    # For each pick station
    for x in fa_PTT.items():
        x = copy(x)
        coeff = coeff.copy()
        # Merge coefficients to fault dataframe
        df = coeff.loc[coeff['Asset Code'].isin(x[1])]
        if len(df)>0:
            df = df.copy()
            # Add pickstation column
            df.loc[:,'PTT'] = str(x[0])
            # Join dataframes together for each pick station
            output = pd.concat([output, df], join='outer', ignore_index=True)

    output.rename(columns={'Asset Code': 'ID', 'Coefficient': 'COEFFICIENT'},
                  inplace=True)
    output.sort_values('PTT', inplace=True)
    return output


@logger.logger
def run_single_model(*, sc_data, weights):
    """
    Summary
    -------
    Runs pre-processing and fitting for model
    ----------
    sc_data: scsdata ScsData
        scs data object
    weights: list
        weighting for hours ([current,1 hour ago, 2 hours ago etc.])
    Returns
    -------
    output: pandas DataFrame
        dataframe containing labeled selected faults
        over reporting window with the number of losts singles
        due to each fault at pick stations
    r2_oos: numeric
        Out of sample R2 score
    num_assets: numeric
        number of selected assets
    Example
    --------
    output, r2_oos, num_assets =run_single_model(sc_data, weights)
    """
    # Creates features set and faults dictionary and transforms
    fa_PTT = sc_data.create_ptt_df(weights=weights)
    sc_data.log_totes()
    # Creates features and target variable and split ready for modelling
    X, y = md.gen_feat_var(sc_data.df, target='Availability',
                           features=['Totes', 'Faults'])
    X_train, X_test, y_train, y_test = \
        md.split(X,
                 y,
                 split_options={'test_size': 0.2,
                                'random_state': None})
    # Catch and ignore NaN OLS warning and runs model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2_oos, coeff, num_assets = md.run_OLS(X_train=X_train,
                                               y_train=y_train,
                                               X_test=X_test,
                                               y_test=y_test, n=30)
    # Creates output dataframe
    output = rs.create_output(fa_PTT, coeff)

    return output, r2_oos, num_assets
