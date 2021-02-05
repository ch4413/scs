# Dependencies
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
import sys
from copy import copy
# Import modules
from scsavailability import results as rs, db, parser as ps, scsdata


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
    # Set run start time
    begin_time = datetime.now()
    # Load source, paths and reoprt window from config
    data_source = config.path.source
    cache_path = r'%scache.csv' % config.path.package
    log_path = r'%srun_log.csv' % config.path.package
    report_start = config.report.start
    report_end = config.report.end

    if data_source == 'Local' or data_source == 'Test':
        # Import local data
        at = pd.read_csv(config.path.totes)
        av = pd.read_csv(config.path.availability)
        fa = pd.read_csv(config.path.faults)
        # Add report dummies to run locally if none set
        if report_start == 'None':
            report_start = pd.to_datetime('2020-01-01 00:00:00', dayfirst=True)
        else:
            report_start = pd.to_datetime(report_start, dayfirst=True)
        if report_end == 'None':
            report_end = pd.to_datetime('2022-01-01 00:00:00', dayfirst=True)
        else:
            report_end = pd.to_datetime(report_end, dayfirst=True)

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

        # Check new data exists if doing an automated run
        if report_start == 'None' and report_end == 'None':
            if fa_max == fa_old_max:
                # Populate run log showing model tried to run but no new scada
                log = pd.read_csv(log_path)
                now = datetime.now()
                runtime = str(now-begin_time)
                timestamp_string = now.strftime("%d-%m-%Y_%H-%M-%S")
                new_row = pd.DataFrame([[timestamp_string,
                                         'No SCADA Data',
                                         'No SCADA Data',
                                         runtime,
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

        # Set report end and start dates
        if report_end == 'None':
            report_end = fa_max.ceil('H')
        else:
            report_end = pd.to_datetime(report_end, dayfirst=True)
        if report_start == 'None':
            report_start = fa_old_max.ceil('H')
        else:
            report_start = pd.to_datetime(report_start, dayfirst=True)

        reporting_window = report_end - report_start

        # Set report start and to lastest 12 hours
        # if cache appears to be out of date
        if reporting_window.days > 7 \
           and config.report.start == 'None' \
           and config.report.end == 'None':
            report_start = report_end - pd.to_timedelta(12, unit='H')
            reporting_window = pd.to_timedelta(12, unit='H')

    # Load pick stations parameters
    speed = config.parameters.speed
    picker_present = config.parameters.picker_present
    availability = config.parameters.availability
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
            output, R2, num_assets = \
                rs.run_single_model(sc_data=sc0,
                                    report_start=report_start,
                                    report_end=report_end,
                                    weights=weights[i],
                                    speed=speed,
                                    picker_present=picker_present,
                                    availability=availability)
        if shift[i] == 15:
            # Run model and produce outputs
            output, R2, num_assets = \
                rs.run_single_model(sc_data=sc15,
                                    report_start=report_start,
                                    report_end=report_end,
                                    weights=weights[i],
                                    speed=speed,
                                    picker_present=picker_present,
                                    availability=availability)
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

    # Read log
    log = pd.read_csv(log_path)
    # Calculate run time
    now = datetime.now()
    runtime = str(now-begin_time)
    # Set model finish timestamp
    timestamp_string = now.strftime("%d-%m-%Y_%H-%M-%S")

    if data_source == 'Test':
        # Create test row
        new_row = pd.DataFrame([[timestamp_string, 'Test', 'Test', runtime,
                                 'Test', 'Test']], columns=log.columns)
    else:
        # Create run metric row
        new_row = pd.DataFrame([[timestamp_string, R2_sel, feat_sel, runtime,
                                 report_start, report_end]],
                               columns=log.columns)
    # Append new row and save run log to folder
    new_log = log.append(new_row, ignore_index=True)
    new_log.to_csv(log_path, index=False)

    # Save output to landing zone
    save_path = r'%sML_output_%s.csv' % (config.path.save, timestamp_string)
    output.to_csv(save_path, index=False)


if __name__ == '__main__':

    print('running with config')
    parser = ArgumentParser(description="Running Pipeline")
    parser.add_argument('--config', required=True,
                        help='path of the YAML file with the configuration')
    args = parser.parse_args()
    config_value = ps.parse_config(args.config)
    run(config_value)
