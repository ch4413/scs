/***
Pulls latest Availability data
***/

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
order by date asc, [Pick Station] asc;