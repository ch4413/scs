# Data Dictionary

## SCS Fault Information

| Field | Description |
|-------|-------------|
|Number |Unique number for a specific fault ID occurring in specific location |
|Alert| String containing location error description |
| Entry Time | Time at which error occurred |
| PLC | Location in the SCS |
| Desk | Sublocation under the PLC |
| Duration | How long was error |
| Fault ID | Identifier for that type of fault |


## Availability Information


| Field | Description | Data Type |Example |
|-------|-------------|-----------|--------|
|Pick Station | Code corresponding to Picking station| Category | PTT999 |
| Datetime | Timestamp at the beginning of availability hour | Timestamp | 01/01/2020 00:00:00 | 
| Availability | Fraction of hour Picking Station was available | Numeric | 0.1|


## Active Totes

| Field | Description | Data Type |Example |
|-------|-------------|-----------|--------|
| ID | Unique code for data collected at module at a given time| Numeric | 99|
| Module| Code for the module | Category | SCS01 | 
| Totes | Number of active blue totes in that module at that point in time | Numeric | 45 |
| Day | Day of the month of recording | Numeric | 9 |
| Month |  Month of recording | Numeric | 11 |
| Year |  Year of recording | Numeric | 2020 |
| Hour |  Hour of recording | Numeric | 8 |
| Minute |  Minute of recording | Numeric | 22 |
