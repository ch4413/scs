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
    fa = pd.read_csv(config.path.faults)
    # Need to fold into pre-processing
    fa = feat.add_code(fa)
    fa, unmapped = feat.add_tote_colour(fa)

    at = feat.pre_process_AT(at)
    av = feat.pre_process_av(av)
    fa = feat.preprocess_faults(fa, remove_same_location_faults = True)

    fa_floor = feat.floor_shift_time_fa(fa, shift=0)

    fa_sel = feat.fault_select(fa_floor, fault_select_options = config.features.fault_select_options)
    
    fa_agg = feat.faults_aggregate(fa_sel,fault_agg_level= config.features.aggregation, agg_type = 'count')

    av_sel,at_sel = feat.av_at_select(av, at, remove_high_AT = True, availability_select_options = None)
    av_agg = feat.aggregate_availability(av_sel, agg_level = config.features.aggregation)
    at_agg = feat.aggregate_totes(at_sel, agg_level = config.features.aggregation)

    df = feat.merge_av_fa_at(av_agg ,at_df=at_agg, fa_df = fa_agg, faults=True, totes = True, agg_level = config.features.aggregation, remove_0=True)

    # Features
    X,y = md.gen_feat_var(df,target = config.model.target)
    y=1-y
    X_train, X_test, y_train, y_test = md.split(X,y)

    # # Model
    Linear_mdl, predictions_LM, Coeff, fit_metrics = md.run_LR_model(X_train, X_test, y_train, y_test)
    X_sel_lm = md.select_features(X, y, Linear_mdl, **config.model.selection_options)
    cv_R2 = md.cross_validate_r2(Linear_mdl, X_sel_lm, y, n_folds = 10, shuffle = True, random_state = 101)

    rt.create_output(config, fit_metrics, Coeff, cv_R2[1])

    print(sc.lm_coefficients(Linear_mdl, X, y))


if __name__ == '__main__':
    print('running with config')

    parser = ArgumentParser(description="Running Pipeline")
    parser.add_argument('--config', required=True,
                        help='path of the YAML file with the configuration')

    args = parser.parse_args()
    config_value = parse_config(args.config)
    
    run(config_value)