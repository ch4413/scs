# Dependencies
import pandas as pd
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
import sys
from copy import copy
# Import modules
from scsavailability import results as rs, db, parser as ps, scsdata, setupdir as sd


def run(config):
    """
    Summary
    -------
    Runs whole model, load and preprocessing data,
    fitting model, and output the results
    ----------
    config: dotmap DotMap
        dotmap obtained from config yaml file parsed
        when module is called
    Returns
    -------
    Example
    --------
    run(config)
    """

    sd.setupdirectory(config.path.package)
    # Set run start time
    begin_time = datetime.now()
    # Load source, mode and paths from config
    data_source = config.source
    mode = config.mode
    print('Running with %s data' % data_source)
    cache_path = r'%scache.csv' % config.path.package
    log_path = r'%srun_log.csv' % config.path.package


    if data_source == 'Local' or data_source == 'Test':
        # Import local data
        at = pd.read_csv(config.path.totes)
        av = pd.read_csv(config.path.availability)
        fa = pd.read_csv(config.path.faults)
        # Save the input data time window
        fa_max = pd.to_datetime(fa['Entry Time'], dayfirst=True).max()
        fa_min = pd.to_datetime(fa['Entry Time'], dayfirst=True).min()
       
    if data_source == 'SQL':
        # Create single connection
        conn = db.mi_db_connection()
        # Query DB using stored package queries
        at_path = r'%sscsavailability/data/sql/active_totes.sql'\
            % config.path.package
        av_path = r'%sscsavailability/data/sql/availability.sql'\
            % config.path.package
        fa_path = r'%sscsavailability/data/sql/faults.sql'\
            % config.path.package

        at = db.read_query(sql_conn=conn, query_path=at_path)
        av = db.read_query(sql_conn=conn, query_path=av_path)
        fa = db.read_query(sql_conn=conn, query_path=fa_path)

        # Read cache and extract max timestamp
        # from current and previous scada data
        fa_old = pd.read_csv(cache_path)
        fa_old_max = pd.to_datetime(fa_old['Entry Time'], dayfirst=True).max()
        fa_max = pd.to_datetime(fa['Entry Time'], dayfirst=True).max()
        fa_min = pd.to_datetime(fa['Entry Time'], dayfirst=True).min()
        if mode == "Automated":
            # Check new data exists if doing an automated run
            if fa_max == fa_old_max:
                # Populate run log showing model tried to run but no new scada
                log = pd.read_csv(log_path)
                if len(log)>0:
                    run_ID = max(log['Run_ID']) + 1
                else:
                    run_ID = 1
                now = datetime.now()
                runtime = str(now-begin_time)
                timestamp_string = now.strftime("%d-%m-%Y_%H-%M-%S")
                new_row = pd.DataFrame([[run_ID,
                                            timestamp_string,
                                            'No SCADA Data',
                                            'No SCADA Data',
                                            runtime,
                                            'No SCADA Data',
                                            'No SCADA Data',
                                            'No SCADA Data']],
                                        columns=log.columns)
                new_log = log.append(new_row, ignore_index=True)
                new_log.to_csv(log_path, index=False)
                # Exit code
                sys.exit('SCADA DATA NOT UPLOADED, MODEL DID NOT RUN')
            else:
                # Save current fault df to cache
                fa.to_csv(cache_path, index=False)

    # Create sc object from Class
    sc = scsdata.ScsData('scs', av, at, fa)
    # Call pre-process methods
    sc.pre_process_at()
    sc.pre_process_av()
    sc.pre_process_faults()
    # Create copy of object for each shift option
    sc0 = copy(sc)
    sc15 = copy(sc)
    # Floor and shift each object
    sc0.floor_shift_time_fa(shift=0)
    sc15.floor_shift_time_fa(shift=15)

    # Set temporal options for looping
    shift = [0, 0, 15, 15]
    weights = [[1], [0.7, 0.2, 0.1], [1], [0.7, 0.2, 0.1]]
    # Set up dictionaries for outputs
    outputs = dict()
    asset_nums = dict()

    for i in range(len(weights)):
        if shift[i] == 0:
            # Run model and produce outputs
            try:
                output, R2, num_assets = \
                    rs.run_single_model(sc_data=sc0, weights=weights[i])
            except:
                print('Error occurred, skipping to next iteration')
                continue
        if shift[i] == 15:
            # Run model and produce outputs
            try:
                output, R2, num_assets = \
                    rs.run_single_model(sc_data=sc0, weights=weights[i])
            except:
                print('Error occurred, skipping to next iteration')
                continue
                
        # Populate output dictionaries
        outputs[R2] = output
        asset_nums[R2] = num_assets

    # Identifiy model with the best fit
    R2_sel = max(k for k, v in outputs.items())
    feat_sel = asset_nums[max(k for k, v in outputs.items())]

    print('Selected R2:', R2_sel)
    print('Number of Selected Assets:', feat_sel)

    # Save output file from model with highest R2
    output = outputs[max(k for k, v in outputs.items())]
    output.loc[:,'TIMESTAMP'] = str(fa_max.ceil('H'))

    # Read log
    log = pd.read_csv(log_path)
    if len(log)>0:
        run_ID = max(log['Run_ID']) + 1
    else:
        run_ID = 1
    # Calculate run time
    now = datetime.now()
    runtime = str(now-begin_time)
    # Set model finish timestamp
    timestamp_string = now.strftime("%d-%m-%Y_%H-%M-%S")

    if data_source == 'Test':
        # Create test row
        new_row = pd.DataFrame([[run_ID, timestamp_string, 'Test', 'Test', runtime, 'Test', 'Test','Test']],
                               columns=log.columns)
    else:
        # Create run metric row
        new_row = pd.DataFrame([[run_ID, timestamp_string, R2_sel, feat_sel, runtime, fa_min.floor('H'), fa_max.ceil('H'), np.nan]],
                               columns=log.columns)
    # Append new row and save run log to folder
    new_log = log.append(new_row, ignore_index=True)
    new_log.to_csv(log_path, index=False)

    # Save output to landing zone
    if data_source=='Test':
        save_path = r'%s/scsavailability/tests/TestData/TestResults/ML_test_output_%d.csv' % (config.path.package, run_ID)
    else:
        save_path = r'%s/outputs/ML_Output_%d.csv' % (config.path.package, run_ID)

    output.to_csv(save_path, index=False)


if __name__ == '__main__':

    print('Running with Config')
    # Load data from config file
    parser = ArgumentParser(description="Running Pipeline")
    parser.add_argument('--config', required=True,
                        help='path of the YAML file with the configuration')
    args = parser.parse_args()
    config_value = ps.parse_config(args.config)
    # Run the model with the config data
    run(config_value)
