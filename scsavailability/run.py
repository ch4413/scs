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
    data_source = config.path.source  # SQL or Local

    if data_source == 'Local':
        # Import local data
        at = pd.read_csv(config.path.totes)
        av = pd.read_csv(config.path.availability,
                         names=["timestamp", "Pick Station",
                                "Availability", "Blue Tote Loss",
                                "Grey Tote Loss"])
        fa = pd.read_csv(config.path.faults)
        # Add report dummies to run locally
        report_start = pd.to_datetime('2020/01/01')
        report_end = pd.to_datetime('2022/01/01')

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
        fa_old = pd.read_csv('./cache.csv')
        fa_old_max = pd.to_datetime(fa_old['Entry Time'], dayfirst=True).max()
        fa_max = pd.to_datetime(fa['Entry Time'], dayfirst=True).max()

        # Check new data exists
        if fa_max == fa_old_max:
            log = pd.read_csv('./Run_log.csv')
            now = datetime.now()
            runtime = str(now-begin_time)
            timestamp_string = now.strftime("%d-%m-%Y_%H-%M-%S")
            new_row = pd.DataFrame([[timestamp_string, 'No SCADA Data',
                                     'No SCADA Data', runtime, 'No SCADA Data',
                                     'No SCADA Data']],
                                   columns=log.columns)
            new_log = log.append(new_row, ignore_index=True)
            new_log.to_csv('./Run_log.csv', index=False)
            sys.exit('SCADA DATA NOT UPLOADED, MODEL DID NOT RUN')
        else:
            fa.to_csv('./cache.csv', index=False)

        report_end = fa_max.ceil('H')
        reporting_window = fa_max.ceil('H') - fa_old_max.ceil('H')

        if reporting_window.days < 2:
            report_start = fa_old_max.ceil('H')
        else:
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

    shift = [0, 0, 15, 15]
    weights = [[1], [0.7, 0.2, 0.1], [1], [0.7, 0.2, 0.1]]
    outputs = dict()
    asset_nums = dict()

    for i in range(len(shift)):
        output, R2, num_assets = rs.run_single_model(sc_data=sc,
                                                     report_start=report_start,
                                                     report_end=report_end,
                                                     shift=shift[i],
                                                     weights=weights[i],
                                                     speed=speed,
                                                     picker_present=picker_present,
                                                     availability=availability)
        outputs[R2] = output
        asset_nums[R2] = num_assets

    R2_sel = max(k for k, v in outputs.items())
    feat_sel = asset_nums[max(k for k, v in outputs.items())]

    print('Selected R2:', R2_sel)
    print('Number of Selected Assets:', feat_sel)

    output = outputs[max(k for k, v in outputs.items())]

    log = pd.read_csv('./Run_log.csv')
    now = datetime.now()
    runtime = str(now-begin_time)
    timestamp_string = now.strftime("%d-%m-%Y_%H-%M-%S")

    new_row = pd.DataFrame([[timestamp_string, R2_sel, feat_sel, runtime,
                             report_start, report_end]], columns=log.columns)
    new_log = log.append(new_row, ignore_index=True)
    new_log.to_csv('./Run_log.csv', index=False)

    path = r'%sML_output_%s.csv' % (config.path.save, timestamp_string)
    output.to_csv(path, index=False)


if __name__ == '__main__':

    print('running with config')
    parser = ArgumentParser(description="Running Pipeline")
    parser.add_argument('--config', required=True,
                        help='path of the YAML file with the configuration')
    args = parser.parse_args()
    config_value = ps.parse_config(args.config)
    run(config_value)
