/***
Pulls latest SCS SCADA Faults data
***/

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
and mindate.mindate < entry_time;