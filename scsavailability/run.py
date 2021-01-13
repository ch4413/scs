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
import sys

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

        at = pd.read_sql(con=mi_db_connection(),sql=                 
                '''
                 SELECT *
FROM [SOLAR].[Data_Power_BI_Active_Totes_SCS] t1 where cast(concat(year,'/',MONTH,'/',DAY) as date) >= getdate()-14
''') 

        av = pd.read_sql(con=mi_db_connection(),sql=
        
        '''
                 with avail as (
select distinct
t.[Pick Station]
,t.date 
,case when g.time is null then 0 else g.time end work_avail_picker_present
,case when a.time is null then 0 else a.time end work_avail_picker_absent
,case when r.time is null then 0 else r.time end work_unvail
from ods.stage.Newton_availability t
left join ods.stage.Newton_availability g on
t.date = g.date
and t.[Pick Station]= g.[Pick Station]
and g.Category = 'Work Avail Picker Present'
left join ods.stage.Newton_availability a on
t.date = a.date
and t.[Pick Station]= a.[Pick Station]
and a.Category = 'Work Avail Picker Absent'
left join ods.stage.Newton_availability r on
t.date = r.date
and t.[Pick Station]= r.[Pick Station]
and r.Category = 'Work Unvail'
)

, merging as (
select
w.date
,w.[Pick Station]
,a.work_avail_picker_absent
,a.work_avail_picker_present
,a.work_unvail
,b.NumberTotes
,b.NumberTotesAdj
from ods.stage.Newton_Waste_Tput w
left join avail a on
a.date = w.date
and a.[Pick Station] = w.[Pick Station]
left join ods.solar.data_analytics_newton_bluetotes_snapshot b on
w.date = b.date
and w.[Pick Station] = b.[Pick Station]
)

, output1 as (
select
merging.*
,case when (work_avail_picker_absent+work_avail_picker_present+work_unvail)=0 then 0 else (work_avail_picker_present+work_avail_picker_absent)/(work_avail_picker_absent+work_avail_picker_present+work_unvail) end as availability
from merging
)

select 
Date as timestamp
,[Pick Station]
,Availability
,case when numbertotesadj is null then null when 1-availability>0.8-0.08*NumberTotesAdj then 0.8-0.08*NumberTotesAdj else 1-availability end as "Blue Tote Loss"
,case when numbertotesadj is null then null when 1-availability>0.8-0.08*NumberTotesAdj then 1-availability-(0.8-0.08*NumberTotesAdj) else 0 end as "Grey Tote Loss"
from output1 o
order by date asc, [Pick Station] asc

''')


        fa = pd.read_sql(con=mi_db_connection(),sql=                 
                '''
                      select Number,Text as Alert,EntryTime as "Entry Time",Group2 as PLC,Group3 as Desk,Duration,"Additional Info 1" as "Fault ID" from SOLAR.ScadaData
    
    where Group2 in('C05',              'C06',     'C07',     'C08',     'C09',     'C10',     'C11',     'C12',     'C13',     'C14',     'C15',     'C16',                'C23',     'C35',     'C36',     'C37',     'C38',     'C39',     'C40',     'C41',     'C42',     'C43',     'C44',     'C45',     'C46',     'C47',                'C48',     'C49',     'C50',     'C51',     'C52',     'SCSM01',            'SCSM02',            'SCSM03',            'SCSM04',                'SCSM05',            'SCSM07',            'SCSM08',            'SCSM09',            'SCSM10',            'SCSM11',            'SCSM12',                'SCSM13',            'SCSM14',            'SCSM15',            'SCSM17',            'SCSM18',            'SCSM19',            'SCSM20')
''') 


    fa_old = pd.read_csv('./cache.csv')

    if fa.equals(fa_old):
        sys.exit('SCADA DATA NOT UPLOADED, MODEL DID NOT RUN') 
    else:
        fa.to_csv('./cache.csv',index=False)    


    speed = config.parameters.speed
    picker_present = config.parameters.picker_present
    availability = config.parameters.availability

    at = feat.pre_process_AT(at)
    av = feat.pre_process_av(av)
    fa, unmapped, end_time = feat.preprocess_faults(fa)

    Shift = [0,0,15,15]
    Weights = [[1],[0.7,0.2,0.1],[1],[0.7,0.2,0.1]]
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
