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
from scsavailability import features as feat, model as md, plotting as pt, results as rt, score as sc

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

    at = pd.read_csv(config.path.totes)
    av = pd.read_csv(config.path.availability)
    scs_raw = pd.read_csv(config.path.faults)

    at = feat.pre_process_AT(at)
    av = feat.pre_process_av(av)
    fa,unmapped = feat.preprocess_faults(scs_raw, remove_same_location_faults = True)

    fa_floor = feat.floor_shift_time_fa(fa, shift=0)

    df,fa_PTT = feat.create_PTT_df(fa_floor,at,av)
    df = feat.log_totes(df) 

    # Features
    X,y = md.gen_feat_var(df, target = config.model.target)
    X_train, X_test, y_train, y_test = md.split(X,y)

    # # Model
    
    R2_cv,R2_OOS,Coeff = md.run_OLS(X_train = X_train,y_train = y_train,X_test = X_test,y_test=y_test, n = 5)
    Output = rt.create_output(fa_PTT,Coeff)

    return Output, R2_cv, R2_OOS

if __name__ == '__main__':
    print('running with config')

    parser = ArgumentParser(description="Running Pipeline")
    parser.add_argument('--config', required=True,
                        help='path of the YAML file with the configuration')

    args = parser.parse_args()
    config_value = parse_config(args.config)
    
    run(config_value)
