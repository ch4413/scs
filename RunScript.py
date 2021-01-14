import pandas as pd
import scsavailability as scs
    
from scsavailability import features as feat, model as md, results as rs


data_source = 'SQL'

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


speed = 470
picker_present = 0.91
availability = 0.71

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

Output.to_csv('\\mshsrmnsukp1405\File Landing Zone\SCADA\Outputs\ML_Output.csv', index = False)

print('Model Ran Successfully and Sent Output File to Landing Zone')
