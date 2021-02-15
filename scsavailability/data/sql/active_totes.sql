/***
Pulls latest Active Totes data
***/

WITH mindate AS (SELECT MAX(convert(datetime,[Entry Time])) maxdate, MAX(convert(datetime,[Entry Time]))-14 mindate FROM SOLAR.newton_AzurePrep_SCADAArchive)

SELECT at.*
FROM SOLAR.newton_AzurePrep_ActiveTotes at
LEFT JOIN mindate on 1=1
WHERE CAST(CONCAT(year,'/',month,'/',day,' ',hour,':',minute,':00') as datetime) > mindate.mindate
AND CAST(CONCAT(year,'/',month,'/',day,' ',hour,':',minute,':00') as datetime) <= mindate.maxdate
ORDER BY CAST(CONCAT(year,'/',month,'/',day,' ',hour,':',minute,':00') as datetime) asc;
