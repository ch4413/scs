import pandas as pd
import scsavailability as scs
import sys
from datetime import datetime
    
from scsavailability import features as feat, model as md, results as rs

begin_time = datetime.now()

data_source = 'SQL'

if data_source == 'Local':

    at = pd.read_csv('C:/Users/Jamie.williams/OneDrive - Newton Europe Ltd/Castle Donnington/Data/active_totes_20201210.csv')
    av = pd.read_csv('C:/Users/Jamie.williams/OneDrive - Newton Europe Ltd/Castle Donnington/Data/Availability_with_Grey&Blue_1811-0912.csv',names = ["timestamp","Pick Station","Availability","Blue Tote Loss","Grey Tote Loss"])
    fa = pd.read_csv('C:/Users/Jamie.williams/OneDrive - Newton Europe Ltd/Castle Donnington/Data/Faults20_11-10_12.csv')

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
                with mindate as (select max(entry_time) maxdate, max(entry_time)-14 mindate from stage.scadadata)

select t.*
from solar.Data_Power_BI_Active_Totes_SCS t
left join mindate on 1=1
where cast(concat(year,'/',month,'/',day,' ',hour,':',minute,':00') as datetime) > mindate.mindate
and cast(concat(year,'/',month,'/',day,' ',hour,':',minute,':00') as datetime) <= mindate.maxdate
order by cast(concat(year,'/',month,'/',day,' ',hour,':',minute,':00') as datetime) asc

''') 

    av = pd.read_sql(con=mi_db_connection(),sql=
    
    '''
                with mindate as (select max(entry_time) maxdate, max(entry_time)-14 mindate from stage.scadadata)

, avail as (
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
from ods.stage.Newton_Waste_Tput w
left join avail a on
a.date = w.date
and a.[Pick Station] = w.[Pick Station]
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
from output1 o
left join mindate on 1=1
where date > mindate.mindate
and date <= mindate.maxdate
order by date asc, [Pick Station] asc

''')


    fa = pd.read_sql(con=mi_db_connection(),sql=                 
            '''
                    with mindate as (select max(entry_time)-14 mindate from stage.scadadata)

select  
Number
,Text as Alert
,ENTRY_TIME as "Entry Time"
,Group2 as PLC
,Group3 as Desk 
,Duration
,Additional_Info_1 as "Fault ID"
from STAGE.ScadaData
left join mindate on 1=1
where Group2 in('C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C23', 'C35', 'C36', 'C37', 'C38', 'C39', 'C40', 'C41', 'C42', 'C43', 'C44', 'C45', 'C46', 'C47', 'C48', 'C49', 'C50', 'C51', 'C52', 'SCSM01', 'SCSM02', 'SCSM03', 'SCSM04', 'SCSM05', 'SCSM07', 'SCSM08', 'SCSM09', 'SCSM10', 'SCSM11', 'SCSM12', 'SCSM13', 'SCSM14', 'SCSM15', 'SCSM17', 'SCSM18', 'SCSM19', 'SCSM20')
and mindate.mindate < entry_time

''') 


fa_old = pd.read_csv('./cache.csv')
fa_old_max = pd.to_datetime(fa_old['Entry Time'],dayfirst=True).max()
fa_max = pd.to_datetime(fa['Entry Time'],dayfirst=True).max()

if fa_max == fa_old_max:
    log = pd.read_csv('./Run_log.csv')
    now = datetime.now()
    runtime = str(now-begin_time)
    timestamp_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    new_row = pd.DataFrame([[timestamp_string,'No SCADA Data','No SCADA Data',runtime,'No SCADA Data','No SCADA Data']],columns = log.columns)
    new_log = log.append(new_row, ignore_index = True)
    new_log.to_csv('./Run_log.csv',index=False)
    sys.exit('SCADA DATA NOT UPLOADED, MODEL DID NOT RUN')
else:
    fa.to_csv('./cache.csv',index=False)    

report_end = fa_max.ceil('H')
reporting_window = fa_max.ceil('H') - fa_old_max.ceil('H')

if reporting_window.days < 2:
    report_start = fa_old_max.ceil('H')
else:
    report_start = fa_max - pd.to_timedelta(12, unit='H')
    reporting_window = pd.to_timedelta(12, unit='H')


speed = 470
picker_present = 0.91
availability = 0.71

at = feat.pre_process_AT(at)
av = feat.pre_process_av(av)
fa, unmapped = feat.preprocess_faults(fa)

Shift = [0,0,15,15]
Weights = [[1],[0.7,0.2,0.1],[1],[0.7,0.2,0.1]]
Outputs = dict()
Asset_Nums = dict()

for i in range(len(Shift)):

    Output, R2, Num_Assets = rs.run_single_model(at,av,fa,report_start,report_end,shift=Shift[i],weights=Weights[i],speed=speed,picker_present=picker_present,availability=availability)

    Outputs[R2] = Output
    Asset_Nums[R2] = Num_Assets

R2_sel = max(k for k, v in Outputs.items())
feat_sel = Asset_Nums[max(k for k, v in Outputs.items())]

print('Selected R2:', R2_sel)
print('Number of Selected Assets:', feat_sel)

Output = Outputs[max(k for k, v in Outputs.items())]

log = pd.read_excel('./Run_log.xlsx')
now = datetime.now()
runtime = str(now-begin_time)
timestamp_string = now.strftime("%d-%m-%Y_%H-%M-%S")

new_row = pd.DataFrame([[timestamp_string,R2_sel,feat_sel,runtime,report_start,report_end]],columns = log.columns)
new_log = log.append(new_row, ignore_index = True)
new_log.to_csv('./Run_log.csv',index=False)
Output.to_csv(r"\\mshsrmnsukp1405\File Landing Zone\SCADA\Outputs\ML_Output_" + "timestamp_string" + ".csv", index = False)