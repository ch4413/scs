# Dependencies
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
import time
from pathlib import Path
import sys
import re
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
                run_ID = re.findall('ML_Output_[0-9]+',str(filename))[0].split('_')[-1]
                load = 1
        else:
            attempt=attempt+1
            if attempt<10:
                time.sleep(300)
            else:
                sys.exit('File Did Not Appear in Outputs Folder')

    timestamp_string = datetime.now()
    save_path = r'%sML_output_%s.csv' % (config.path.save, timestamp_string)
    coeff.to_csv(save_path, index=False)


if __name__ == '__main__':

    print('Exporting Output File')
    parser = ArgumentParser(description="Running Pipeline")
    parser.add_argument('--config', required=True,
                        help='path of the YAML file with the configuration')
    args = parser.parse_args()
    config_value = ps.parse_config(args.config)
    export(config_value)