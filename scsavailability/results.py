import pandas as pd
import numpy as np
from . import logger
import warnings
from scsavailability import model as md, results as rs


@logger.logger
def create_output(fa_PTT, Coeff, report_start, report_end, speed=470,
                  picker_present=0.91, availability=0.71):
    """
    Summary
    -------
    Create output dataframe of lost single due
    to faults over reporting window from
    coefficients and pick station faults dictionary
    ----------
    fa_PTT: dictionary
        dictionary of pick station specific fault dataframes
    Coeff: pandas DataFrame
        Selected assets and coefficients from ML Model
    report start: datetime object
        reporting window start
    report end: datetime object
        reporting window end
    speed: numeric
        average picker speed
    picker_present: numeric
        average pick present
    availability: numeric
        average availability
    Returns
    -------
    final_output: pandas DataFrame
        dataframe containing labeled selected faults
        over reporting window with the number of losts singles
        due to each fault at pick stations
    Example
    --------
    output = create_output(fa_PTT,
                           Coeff,
                           report_start,
                           report_end,
                           speed=470,
                           picker_present=0.91,
                           availability=0.71)
    """
    # Create empty dataframe to fill
    Output = pd.DataFrame(columns=['Alert ID', 'Alert', 'Fault ID',
                                   'Asset Code', 'Tote Colour',
                                   'Quadrant', 'MODULE', 'Entry Time'])

    # For each pick station
    for x in fa_PTT.items():
        # Merge coefficients to fault dataframe
        df = x[1].merge(Coeff, how="inner", on="Asset Code")
        # Calculate downtime due to fault
        df['Downtime'] = abs(df['Coefficient']) * df['Duration']
        # Create ones and pick station columns
        df['ones'] = pd.Series(np.ones(len(df)))
        df['PTT'] = str(x[0])

        # Calculate singles lost by taking product of downtime
        # and pick station parameters
        # Minimum of 1 and donwtime taken as downtime can't exceed 100%
        df['Singles'] = df[['Downtime', 'ones']].min(axis=1)\
            * (speed * picker_present * availability)
        # Drop columns not required in output
        df.drop(['timestamp', 'Duration', 'Loop', 'Suffix', 'PLCN',
                 'Alert Type', 'Pick Station', 'Coefficient', 'Downtime',
                 'ones'], axis=1, inplace=True)
        # Join dataframes together for each pick station
        Output = pd.concat([Output, df], join='outer', ignore_index=True)
    Output.fillna(0, inplace=True)

    # Only select rows in reporting window
    Output = Output[(Output['Entry Time'] >= report_start)
                    & (Output['Entry Time'] < report_end)]

    # Rename columns to allow them to be stored easily in SQL
    final_output = pd.DataFrame({'NUMBER': Output['Number'],
                                 'ALERT': Output['Alert'],
                                 'ENTRY_TIME': Output['Entry Time'],
                                 'END_TIME': Output['End Time'],
                                 'PLC': Output['PLC'],
                                 'DESK': Output['Desk'],
                                 'FAULT_ID': Output['Fault ID'],
                                 'ID': Output['Asset Code'],
                                 'AREA': Output['Area'],
                                 'BLUEGREY': Output['Tote Colour'],
                                 'PTT': Output['PTT'],
                                 'SINGLES': Output['Singles']
                                 })

    return final_output


@logger.logger
def run_single_model(*, sc_data, report_start, report_end, weights,
                     speed, picker_present, availability):
    """
    Summary
    -------
    Runs pre-processing and fitting for model
    ----------
    sc_data: scsdata ScsData
        scs data object
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
    output, r2_oos, num_assets =run_single_model(sc_data,
                                                 report_start,
                                                 report_end,
                                                 weights,
                                                 speed,
                                                 picker_present,
                                                 availability)
    """
    # Creates features set and faults dictionary and transforms
    fa_PTT = sc_data.create_ptt_df(weights=weights)
    sc_data.log_totes()
    # Creates features and target variable and split ready for modelling
    X, y = md.gen_feat_var(sc_data.df, target='Availability',
                           features=['Totes', 'Faults'])
    X_train, X_test, y_train, y_test = md.split(X,
                                                y,
                                                split_options={'test_size': 0.2,
                                                               'random_state': None})
    # Catch and ignore NaN OLS warning and runs model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2_oos, coeff, num_assets = md.run_OLS(X_train=X_train, y_train=y_train,
                                               X_test=X_test, y_test=y_test, n=30)
    # Creates output dataframe
    output = rs.create_output(fa_PTT, coeff, report_start, report_end,
                              speed=speed, picker_present=picker_present,
                              availability=availability)

    return output, r2_oos, num_assets
