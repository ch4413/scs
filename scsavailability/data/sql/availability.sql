/***
Pulls latest Availability data
***/

with mindate as (select max(convert(datetime,[Entry Time])) maxdate, max(convert(datetime,[Entry Time]))-14 mindate from SOLAR.newton_AzurePrep_SCADAArchive)

, avail as (
select distinct
t.PTT [Pick Station]
,t.hour date
,case when g.time_allocated is null then 0 else g.time_allocated end work_avail_picker_present
,case when a.time_allocated is null then 0 else a.time_allocated end work_avail_picker_absent
,case when r.time_allocated is null then 0 else r.time_allocated end work_unvail
from SOLAR.newton_AzurePrep_Availability t
left join SOLAR.newton_AzurePrep_Availability g on
t.hour = g.hour
and t.PTT = g.PTT
and g.time_cat = 'Work Avail Picker Present'
left join SOLAR.newton_AzurePrep_Availability a on
t.hour = a.hour
and t.PTT = a.PTT
and a.time_cat = 'Work Avail Picker Absent'
left join SOLAR.newton_AzurePrep_Availability r on
t.hour = r.hour
and t.PTT = r.PTT
and r.time_cat = 'Work Unvail'
)

, output as (
select
avail.*
,case	when	(work_avail_picker_absent+work_avail_picker_present+work_unvail) = 0
		then	0 
		else	(work_avail_picker_present+work_avail_picker_absent)/(work_avail_picker_absent+work_avail_picker_present+work_unvail)
		end		availability
from avail
)

select 
Date as timestamp
,[Pick Station]
,Availability
from output o
left join mindate on 1=1
where date > mindate.mindate
and date <= mindate.maxdate
order by date asc, [Pick Station] asc;