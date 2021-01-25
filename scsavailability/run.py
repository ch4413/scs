# Dependencies
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
import sys
import pkg_resources as pkg
# Import modules
from scsavailability import results as rs, db, parser as ps, scsdata


def run(config):
    """
    Summary
    -------
    Runs pre-processing and fitting for model
    ----------
    config: dotmap DotMap
    Returns
    -------
    Example
    --------
    run(config)
    """
    begin_time = datetime.now()
    data_source = config.path.source # SQL or Local
    cache_path = r'%scache.csv' % config.path.package
    log_path = r'%srun_log.csv' % config.path.package
    report_start = config.report.start
    report_end = config.report.end


    if data_source == 'Local':
        # Import local data
        at = pd.read_csv(config.path.totes)

        av = pd.read_csv(config.path.availability)
        fa = pd.read_csv(config.path.faults)
        # Add report dummies to run locally if none set
        if report_start == 'None':
            report_start = pd.to_datetime('2020-01-01 00:00:00',dayfirst=True)
        else:
            report_start = pd.to_datetime(report_start,dayfirst=True)    
        if report_end == 'None':    
            report_end = pd.to_datetime('2022-01-01 00:00:00',dayfirst=True)
        else:
            report_end = pd.to_datetime(report_end,dayfirst=True)    

    if data_source == 'SQL':
        # Create single connection
        conn = db.mi_db_connection()
        # Query DB using stored package queries
        at = pd.read_sql(con=conn, sql=pkg.resource_stream(__name__,
                         'data/sql/active_totes.sql'))
        av = pd.read_sql(con=conn, sql=pkg.resource_stream(__name__,
                         'data/sql/availability.sql'))
        fa = pd.read_sql(con=conn, sql=pkg.resource_stream(__name__,
                         'data/sql/active_totes.sql'))
        # Read cache

        fa_old = pd.read_csv(cache_path)
        fa_old_max = pd.to_datetime(fa_old['Entry Time'],dayfirst=True).max()
        fa_max = pd.to_datetime(fa['Entry Time'],dayfirst=True).max()

        # Check new data exists
        if report_start == 'None' and report_end == 'None':   
            if fa_max == fa_old_max:
                log = pd.read_csv(log_path)
                now = datetime.now()
                runtime = str(now-begin_time)
                timestamp_string = now.strftime("%d-%m-%Y_%H-%M-%S")
                new_row = pd.DataFrame([[timestamp_string,'No SCADA Data',
                                         'No SCADA Data',runtime, 'No SCADA Data',
                                         'No SCADA Data']],
                                          columns = log.columns)
                new_log = log.append(new_row, ignore_index = True)
                new_log.to_csv(log_path,index=False)
                sys.exit('SCADA DATA NOT UPLOADED, MODEL DID NOT RUN')
            else:
                fa.to_csv(cache_path,index=False)    

        if report_end == 'None':    
            report_end = fa_max.ceil('H')
        else:
            report_end = pd.to_datetime(report_end,dayfirst=True)      
        if report_start == 'None':    
            report_end = fa_old_max.ceil('H')
        else:
            report_start = pd.to_datetime(report_start,dayfirst=True)    

        reporting_window = report_end - report_start

        if reporting_window.days > 7 and config.report.start == 'None' and config.report.end == 'None':
            report_start = report_end - pd.to_timedelta(12, unit='H')
            reporting_window = pd.to_timedelta(12, unit='H')

    speed = config.parameters.speed
    picker_present = config.parameters.picker_present
    availability = config.parameters.availability
    # Create sc object from Class
    sc = scsdata.ScsData('scs', av, at, fa)
    # Call pre-process methods
    sc.pre_process_at()
    sc.pre_process_av()
    sc.pre_process_faults()
    sc0 = sc
    sc15 = sc
    sc0.floor_shift_time_fa(shift=0)
    sc15.floor_shift_time_fa(shift=15)

    shift = [0, 0, 15, 15]
    weights = [[1], [0.7, 0.2, 0.1], [1], [0.7, 0.2, 0.1]]
    outputs = dict()
    asset_nums = dict()

    for i in range(len(weights)):
        if shift[i] == 0:
            output, R2, num_assets = rs.run_single_model(sc_data=sc0, 
                                                         report_start=report_start,
                                                         report_end=report_end,
                                                         weights=weights[i],
                                                         speed=speed, 
                                                         picker_present=picker_present,
                                                         availability=availability)
        if shift[i] == 15:    
             output, R2, num_assets =rs.run_single_model(sc_data=sc15, 
                                                         report_start=report_start,
                                                         report_end=report_end,
                                                         weights=weights[i],
                                                         speed=speed, 
                                                         picker_present=picker_present,
                                                         availability=availability)

        outputs[R2] = output
        asset_nums[R2] = num_assets
        print(output['ENTRY_TIME'])

    R2_sel = max(k for k, v in outputs.items())
    feat_sel = asset_nums[max(k for k, v in outputs.items())]

    print('Selected R2:', R2_sel)
    print('Number of Selected Assets:', feat_sel)

    output = outputs[max(k for k, v in outputs.items())]

    log = pd.read_csv(log_path)
    now = datetime.now()
    runtime = str(now-begin_time)
    timestamp_string = now.strftime("%d-%m-%Y_%H-%M-%S")


    new_row = pd.DataFrame([[timestamp_string,R2_sel,feat_sel,runtime,
                             report_start,report_end]],columns = log.columns)
    new_log = log.append(new_row, ignore_index = True)
    new_log.to_csv(log_path,index=False)

    save_path = r'%sML_output_%s.csv' % (config.path.save,timestamp_string)
    output.to_csv(save_path, index = False)


if __name__ == '__main__':

    print('running with config')
    parser = ArgumentParser(description="Running Pipeline")
    parser.add_argument('--config', required=True,
                        help='path of the YAML file with the configuration')
    args = parser.parse_args()
    config_value = ps.parse_config(args.config)
    run(config_value)
