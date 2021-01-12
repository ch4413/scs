### Run Script
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from dotmap import DotMap
import os
import re
import yaml

import scsavailability as scs
from scsavailability import features as feat, model as md, results as rs

def parse_config(path=None, data=None, tag='!ENV'):
    """
    Load a yaml configuration file and resolve any environment variables
    The environment variables must have !ENV before them and be in this format
    to be parsed: ${VAR_NAME}.
    E.g.:
    database:
        host: !ENV ${HOST}
        port: !ENV ${PORT}
    app:
        log_path: !ENV '/var/${LOG_PATH}'
        something_else: !ENV '${AWESOME_ENV_VAR}/var/${A_SECOND_AWESOME_VAR}'
    :param str path: the path to the yaml file
    :param str data: the yaml data itself as a stream
    :param str tag: the tag to look for
    :return: the dict configuration
    :rtype: dict[str, T]
    """
    # pattern for global vars: look for ${word}
    pattern = re.compile(r'.*?\${(\w+)}.*?')
    loader = yaml.SafeLoader

    # the tag will be used to mark where to start searching for the pattern
    # e.g. somekey: !ENV somestring${MYENVVAR}blah blah blah
    loader.add_implicit_resolver(tag, pattern, None)

    def constructor_env_variables(loader, node):
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        if match:
            full_value = value
            for g in match:
                full_value = full_value.replace(
                    f'${{{g}}}', os.environ.get(g, g)
                )
            return full_value
        return value

    loader.add_constructor(tag, constructor_env_variables)

    if path:
        with open(path) as conf_data:
            return DotMap(yaml.load(conf_data, Loader=loader))
    elif data:
        return DotMap(yaml.load(data, Loader=loader))
    else:
        raise ValueError('Either a path or data should be defined as input')


def run(config):
    """

    :param config:
    :return:
    """
    
    data_source = config.path.source

    if data_source == 'Local':

        at = pd.read_csv(config.path.totes)
        av = pd.read_csv(config.path.availability,names = ["timestamp","Pick Station","Availability","Blue Tote Loss","Grey Tote Loss"])
        fa = pd.read_csv(config.path.faults)

    if data_source == 'SQL':

        def mi_db_connection(): 
            import pyodbc
            conn = pyodbc.connect('Driver={SQL Server};'
                            'Server=MSHSRMNSUKP1405;'
                            'Database=ODS;'
                            'as_dataframe=True')
            return conn

        at = pd.read_sql(con=mi_db_connection(),sql=config.path.totes)
        av = pd.read_sql(con=mi_db_connection(),sql=config.path.availability)
        fa = pd.read_sql(con=mi_db_connection(),sql=config.path.faults)

    speed = config.parameters.speed
    picker_present = config.parameters.picker_present
    availability = config.parameters.availability

    print(speed)
    print(picker_present)
    print(availability)

    at = feat.pre_process_AT(at)
    av = feat.pre_process_av(av)
    fa, unmapped, end_time = feat.preprocess_faults(fa)

    Shift = [0]#[0,0,0,10,10,10,20,20,20]
    Weights = [1]#[[1],[0.7,0.3],[0.7,0.2,0.1],[1],[0.7,0.3],[0.7,0.2,0.1],[1],[0.7,0.3],[0.7,0.2,0.1]]
    Outputs = dict()

    for i in range(len(Shift)):

        Output, R2 = rs.run_single_model(at,av,fa,end_time,shift=Shift[i],weights=Weights[i],speed=speed,picker_present=picker_present,availability=availability)

        Outputs[R2] = Output

    print('Selected R2:', max(k for k, v in Outputs.items()))

    Output = Outputs[max(k for k, v in Outputs.items())]

    Output.to_csv(config.path.save, index = False)


if __name__ == '__main__':
    print('running with config')

    parser = ArgumentParser(description="Running Pipeline")
    parser.add_argument('--config', required=True,
                        help='path of the YAML file with the configuration')

    args = parser.parse_args()
    config_value = parse_config(args.config)
    
    run(config_value)
