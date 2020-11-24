# Data Dictionary

## SCS Fault Information

| Field | Description |
|-------|-------------|
|Number |Unique number for a specific fault ID occurring in specific location |
|Text | String containing location error description |
| Entry Time | Time at which error occurred |
| PLC | Location in the SCS |
| Desk | Sublocation under the PLC |
| Duration | How long was error |
| Fault ID | Identifier for that type of fault |


## Availability Information


| Field | Description | Data Type |Example |
|-------|-------------|--|--|
|Pick Station | Code corresponding to Picking station| Category | PTT999 |
| Datetime | Timestamp at the beginning of availability hour | Timestamp | 01/01/2020 00:00:00 | 
| Availability | Fraction of hour Picking Station was available | Numeric | 0.1|

