# SCS Availability

## Project Aim

1. Create model that shows good relationship between `SCS Alerts` data and `availability` data at a:

* SCS level
* Quadrant level
* Module level
* Pick station level

**Note:** we want a model that explains the relationship over its predicting power.

2. Return most important features from model.
* Which features are have strongest impact on model?
* How much time is associated with each downtime?

*"Error X has the largest influence on availability/downtime"* or *"Error Y contributes X minutes to downtime per hour"*

3. Report "most important" faults/errors to engineers on serviceable dashboard.

Model and data considerations:

* Where will the data come from and be stored? At what frequency?
* Where will the pipeline and model be running from? What infrastructure is there currently? i.e. server with python on, job scheduler, API for calling
* How will we model and maintain good model performance in the event of a data/environment change?

DS: data refresh is currently every 12 hours, with an aim to have it refreshing every 15 minutes. We are currently thinking of re-fitting the model for each new data.

## Current state

* Engineers repair/fix machines based on most frequent error messages or their experience.
* They may not be fixing the most important faults. Some infrequent faults lead to greater primary or secondary downtime.

Independently from this, we are creating a rules-based engine for associating faults and downtime. This is based on studying the line.

## Data

Data is stored on the Sharepoint site of M&S in the Data directory. We will start by focussing on the `availability` and `SCS Alerts` data.
* `availability` is for October and November 2020
* `SCS Alerts` for 

### Getting started

Download the data and code

```
git clone https://github.com/ch4413/scs
```

Data is stored on the Sharepoint site of M&S in the Data directory.

### Contact

* [Dan Simpson](dan.simpson@newtoneurope.com)
* [Jamie Williams](Jamie.Williams@newtoneurope.com)
* [Christopher Hughes](chris.hughes@newtoneurope.com)