/***
Pulls latest Active Totes data
***/

WITH mindate AS (SELECT MAX(entry_time) maxdate, MAX(entry_time)-14 mindate FROM stage.scadadata)

SELECT at.*
FROM solar.Data_Power_BI_Active_Totes_SCS at
LEFT JOIN mindate on 1=1
WHERE CAST(CONCAT(year,'/',month,'/',day,' ',hour,':',minute,':00') as datetime) > mindate.mindate
AND CAST(CONCAT(year,'/',month,'/',day,' ',hour,':',minute,':00') as datetime) <= mindate.maxdate
ORDER BY CAST(CONCAT(year,'/',month,'/',day,' ',hour,':',minute,':00') as datetime) asc;
