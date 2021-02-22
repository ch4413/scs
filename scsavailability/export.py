# Dependencies
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
import time
from pathlib import Path
import sys
import os
import re
import shutil
# Import modules
from scsavailability import db, parser as ps, setupdir as sd

# Set number of threads to stop warning
os.environ['NUMEXPR_NUM_THREADS'] = '8'


def export(config):

    """
    Summary
    -------
    Loads output file from results folder, exports to
    landing zone or SQL depending on config option, updated
    run log with export time and move file to archive
    ----------
    config: dotmap DotMap
        dotmap obtained from config yaml file parsed
        when module is called
    Returns
    -------
    Example
    --------
    export(config)
    """
    # Check if directory set up correctly
    sd.setupdirectory(config.path.package)
    # Load path and source
    log_path = r'%srun_log.csv' % config.path.package
    output_path = r'%s\outputs' % config.path.package
    data_source = config.source

    # Set count and flag
    load = 0
    attempt = 0
    while load == 0:
        # Whilst no file has been loaded, check if a file with
        # ML_Output_x format is in outputs folder
        if list(Path(output_path).glob("ML_Output_*.csv")):
            for filename in Path(output_path).glob("*.csv"):
                # For each file found, read the data
                # and move to the archive folder
                coeff = pd.read_csv(filename)
                shutil.move(str(filename), config.path.package
                            + 'outputs/Archive')
                # Extract Run ID from file name
                run_ID = re.findall('ML_Output_[0-9]+',
                                    str(filename))[0].split('_')[-1]
                print('Attempting to Export Run %s Output' % run_ID)
                if data_source == 'Local' or data_source == 'Test':
                    # If exporting to local folder, load current time
                    now = datetime.now()
                    timestamp_string = now.strftime("%d-%m-%Y_%H-%M-%S")
                    # Set save path from config and save
                    # file in folder with timestamp in name
                    save_path = r'%sML_Output_%s.csv'\
                        % (config.path.save, timestamp_string)
                    coeff.to_csv(save_path, index=False)
                elif data_source == 'SQL':
                    # If exporting direct to SQL, load output query
                    query_path = r'%sscsavailability/data/sql/coeff.sql'\
                        % config.path.package
                    # Create connection
                    conn = db.mi_db_connection()
                    # Load coefficient old table
                    old_sql = db.read_query(sql_conn=conn,
                                            query_path=query_path)
                    # Append new values
                    new_sql = old_sql.append(coeff, ignore_index=False)
                    # Remove any old duplicates of the new coefficient values
                    new_sql.drop_duplicates(subset=['ID','PTT'],
                                            keep='last',
                                            inplace=True)
                    # Remove any coefficients that are more than 2 months old
                    date_thres = pd.to_datetime("today")\
                        - pd.to_timedelta(60, unit='days')
                    new_sql = new_sql[pd.to_datetime(new_sql['TIMESTAMP'],
                                                     dayfirst=True)
                                      > date_thres]
                    # Sort and replace old table with new values in SQL
                    new_sql.sort_values('PTT', inplace=True)
                    db.output_to_sql(
                        output=new_sql,
                        sql_conn=conn,
                        tablename='newton_AzurePrep_MLCoefficients',
                        schema='SOLAR')
                else:
                    log = pd.read_csv(log_path)
                    # Set export time to unsuccesful
                    # and write table back to folder
                    log['Export_Time'].fillna('Export Unsuccessful',
                                              inplace=True)
                    log.to_csv(log_path, index=False)
                    # Exit code with error message
                    sys.exit('Invalid Source')

                # If successfully exported, load log
                log = pd.read_csv(log_path)
                # Fill in export time for row matching
                # Run ID extracted from file
                now = datetime.now()
                timestamp_string = now.strftime("%d-%m-%Y_%H-%M-%S")
                log.loc[log['Run_ID'] == int(run_ID),
                        'Export_Time'] = timestamp_string
                # Write log back to scs folder
                log.to_csv(log_path, index=False)
                print('Run %s Output Exported Successfully' % run_ID)
            # Set load flag to 1 to exit loop
            load = 1
        else:
            # If no file in outputs
            attempt = attempt+1
            if attempt < 10:
                print('No file matching format, attempted to load %d times,'
                      ' will try again in 5 mins' % attempt)
                # If less than 10 attempts, wait 5 minutes and try again
                time.sleep(300)
            else:
                # Once 10 attempts have been tried, load log
                log = pd.read_csv(log_path)
                # Set export time to unsuccesful and write table back to folder
                log['Export_Time'].fillna('Export Unsuccessful', inplace=True)
                log.to_csv(log_path, index=False)
                # Exit code with error message
                sys.exit('File Did Not Appear in Outputs Folder')


if __name__ == '__main__':

    print('Exporting Output File')
    # Loading data from config
    parser = ArgumentParser(description="Running Pipeline")
    parser.add_argument('--config', required=True,
                        help='path of the YAML file with the configuration')
    args = parser.parse_args()
    config_value = ps.parse_config(args.config)
    # Run export function
    export(config_value)
