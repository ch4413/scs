# Dependencies
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
import time
from pathlib import Path
import sys
import re
import shutil
# Import modules
from scsavailability import db, parser as ps

def export(config):

    log_path = r'%srun_log.csv' % config.path.package
    data_source = config.path.source

    load = 0
    attempt = 0
    while load==0:
        if list(Path(r"C:\Users\Jamie.williams\Desktop\scs\outputs").glob("ML_Output_*.csv")):
            for filename in Path(r"C:\Users\Jamie.williams\Desktop\scs\outputs").glob("*.csv"):
                coeff = pd.read_csv(filename)
                shutil.move(str(filename), config.path.package + 'outputs/Archive')
                run_ID = re.findall('ML_output_[0-9]+',str(filename))[0].split('_')[-1]
                if data_source == 'Local':
                    now = datetime.now()
                    timestamp_string = now.strftime("%d-%m-%Y_%H-%M-%S")
                    save_path = r'%sML_output_%s.csv' % (config.path.save, timestamp_string)
                    coeff.to_csv(save_path, index=False)
                elif data_source == 'SQL':
                    #insert how to write using odbc
                    query_path = r'%sscsavailability/data/sql/coeff.sql'\
                    % config.path.package
                    conn = db.mi_db_connection()
                    old_sql = db.read_query(sql_conn=conn, query_path=query_path)
                    new_sql = old_sql.append(coeff, ignore_index=False)
                    new_sql.drop_duplicates(
                    subset='ID',
                    keep='last')
                    date_thres = today = pd.to_datetime("today") - pd.to_timedelta(60,unit='days')
                    new_sql = new_sql[pd.to_datetime(new_sql['TIMESTAMP'],day_first=True) > date_thres]
                    new_sql.sort_values('PTT', inplace=True)
                    new_sql.to_sql('SOLAR.newton_AzurePrep_MLCoefficients',conn,if_exists='replace',index=False)

                load = 1
        else:
            attempt=attempt+1
            if attempt<10:
                time.sleep(300)
            else:
                log = pd.read_csv(log_path)
                log['Export_time'].fillna('Export Unsuccessful',inplace=True)
                log.to_csv(log_path, index=False)
                sys.exit('File Did Not Appear in Outputs Folder')

    log = pd.read_csv(log_path)
    log.loc[log['Run_ID']==int(run_ID),'Export_time'] = timestamp_string
    log.to_csv(log_path, index=False)


if __name__ == '__main__':

    print('Exporting Output File')
    parser = ArgumentParser(description="Running Pipeline")
    parser.add_argument('--config', required=True,
                        help='path of the YAML file with the configuration')
    args = parser.parse_args()
    config_value = ps.parse_config(args.config)
    export(config_value)