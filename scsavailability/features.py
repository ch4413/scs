# Import required packages

import pandas as pd
import numpy as np
from . import logger
import os
import pkg_resources

# Set number of threads to stop warning
os.environ['NUMEXPR_NUM_THREADS'] = '8'


@logger.logger
def load_module_lookup():
    """
    Summary
    -------
    Loads module lookup table to map PLC and Desk Codes to Modules
    Returns
    -------
    module_lu: pandas DataFrame
        dataframe containing module mapping
    Example
    --------
    lu = load_module_lookup()
    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream(__name__, 'data/module_lookup.csv')
    module_lu = pd.read_csv(stream)
    return module_lu


@logger.logger
def load_tote_lookup():
    """
    Summary
    -------
    Loads tote lookup which maps asset codes to tote colours as scs locations
    Returns
    -------
    tote_lu: pandas DataFrame
        dataframe containing asset mapping
    Example
    --------
    lu = load_tote_lookup()
    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream(__name__, 'data/tote_lookup.csv')
    tote_lu = pd.read_csv(stream)
    return tote_lu


@logger.logger
def load_ID_lookup():
    """
    Summary
    -------
    Loads tote lookup which defines Fault IDs as warnings or faults
    Returns
    -------
    id_lu: pandas DataFrame
        dataframe containing Fault ID categorisation
    Example
    --------
    lu = load_ID_lookup()
    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream(__name__, 'data/ID_lookup.csv')
    id_lu = pd.read_csv(stream)
    return id_lu


@logger.logger
def load_PTT_lookup():
    """
    Summary
    -------
    Loads PTT lookup which identifies pick station specific assets
    Returns
    -------
    ptt_lu: pandas DataFrame
        dataframe containing mapping for PTT specific assets
    Example
    --------
    lu = load_PTT_lookup()
    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream(__name__, 'data/PTT_lookup.csv')
    ptt_lu = pd.read_csv(stream)
    return ptt_lu


@logger.logger
def pre_process_at(active_totes):
    """
    Summary
    -------
    Processes the active totes dataframe converting columns to datetimes
    and mapping modules to quadrants
    Parameters
    ----------
    active_totes: pandas DataFrame
        raw active totes data
    Returns
    -------
    active_totes: pandas DataFrame
        processed active totes dataframe
    Example
    --------
    at = pre_process_at(active_totes)
    """
    # Remove ECB and RCB rows
    active_totes = active_totes[~active_totes['MODULE_ASSIGNED'].isin(['ECB', 'RCB'])].copy()

    # Extracting module number
    active_totes['MODULE_ASSIGNED'] = active_totes['MODULE_ASSIGNED'].apply(lambda x: x[3:])

    # Convert module number from string into number
    active_totes['MODULE_ASSIGNED'] = active_totes['MODULE_ASSIGNED'].apply(pd.to_numeric)

    # Convert time and date data to string to allow convertion to datetime object
    active_totes['DAY'] = active_totes['DAY'].astype('str').str.pad(width=2,
                                                                    side='left',
                                                                    fillchar='0')
    active_totes['HOUR'] = active_totes['HOUR'].astype('str').str.pad(width=2,
                                                                      side='left',
                                                                      fillchar='0')
    active_totes['MINUTE'] = active_totes['MINUTE'].astype('str').str.pad(width=2,
                                                                          side='left',
                                                                          fillchar='0')

    # Create timestamp for each row from the time and date columns
    active_totes['timestamp'] = pd.to_datetime(active_totes.apply(
                    lambda x: '{0}/{1}/{2} {3}:{4}'.format(x['DAY'],
                                                           x['MONTH'],
                                                           x['YEAR'],
                                                           x['HOUR'],
                                                           x['MINUTE']),
                    axis=1),
                                               dayfirst=True)

    # Drop old date and time columns
    active_totes = active_totes.drop(['DAY', 'MONTH', 'YEAR',
                                      'HOUR', 'MINUTE', 'ID'], axis=1)
    active_totes.rename(columns={'MODULE_ASSIGNED': 'Module'},
                        inplace=True)

    # Creating quadrant column based off module mapping
    active_totes['Quadrant'] = 0
    active_totes.loc[active_totes['Module'] < 6, 'Quadrant'] = 1
    active_totes.loc[(active_totes['Module'] > 6)
                     & (active_totes['Module'] < 11), 'Quadrant'] = 2
    active_totes.loc[(active_totes['Module'] > 10)
                     & (active_totes['Module'] < 16), 'Quadrant'] = 3
    active_totes.loc[(active_totes['Module'] > 16)
                     & (active_totes['Module'] < 21), 'Quadrant'] = 4
    return active_totes


@logger.logger
def pre_process_av(av):
    """
    Summary
    -------
    Processes the availability dataframe converting timestamp to datetime
    and mapping rows to quadrants and modules
    Parameters
    ----------
    av: pandas DataFrame
        raw availability data
    Returns
    -------
    av: pandas DataFrame
        processed availability dataframe
    Example
    --------
    av = pre_process_av(av)
    """
    # Rename time column
    av.rename(columns={av.columns[0]: 'timestamp'}, inplace=True)

    # Assign Pick Station to Quadrant
    Quad_1 = ['PTT011', 'PTT012', 'PTT021', 'PTT022', 'PTT031', 'PTT032',
              'PTT041', 'PTT042', 'PTT051', 'PTT052']
    Quad_2 = ['PTT071', 'PTT072', 'PTT081', 'PTT082', 'PTT091', 'PTT092',
              'PTT101', 'PTT102']
    Quad_3 = ['PTT111', 'PTT112', 'PTT121', 'PTT122', 'PTT131', 'PTT132',
              'PTT141', 'PTT142', 'PTT151', 'PTT152']
    Quad_4 = ['PTT171', 'PTT172', 'PTT181', 'PTT182', 'PTT191', 'PTT192',
              'PTT201', 'PTT202']

    # Assign availability to Quadrants
    Quad = []
    for i in av['Pick Station']:
        if i in Quad_1:
            Quad.append(1)
        elif i in Quad_2:
            Quad.append(2)
        elif i in Quad_3:
            Quad.append(3)
        elif i in Quad_4:
            Quad.append(4)
    av['Quadrant'] = Quad

    # Assign availability to Modules
    av['Module'] = (av['Pick Station'].str[3].astype(int)*10 +
                    av['Pick Station'].str[4].astype(int))
    av['timestamp'] = pd.to_datetime(av['timestamp'], dayfirst=True)
    return(av)


@logger.logger
def add_code(data):
    """
    Summary
    -------
    Extracts asset code from alert string in faults dataset
    Parameters
    ----------
    data: pandas DataFrame
        dataframe of features
    Returns
    -------
    scs: pandas DataFrame
        dataframe with 'code'
    Example
    --------
    scs = add_code(data)
    """
    scs = data.copy()
    # Extracts PLC Numbers
    scs['PLCN'] = scs['PLC'].str.extract(
        r'((?<=C)[0-9]{2})').fillna(0).astype('int')
    # Asset code matching patterns extracted from Alert Columns
    scs['Asset Code'] = scs['Alert'].str.extract(
        r'(^[A-Z]{3}[0-9]{3}|[A-Z][0-9]{4}[A-Z]{3}[0-9]{3}|[A-Z]{3} [A-Z][0-9]{2})')
    scs['Asset Code'] = scs['Asset Code'].str.replace(" ", "")
    # Extract PTT asset codes
    scs.loc[scs['Alert'].str.contains('PTT'), 'Asset Code'] = scs.loc[scs['Alert'].str.contains('PTT')]['Alert'].str.extract(r'(C[0-9]{2}PTT[0-9]{3})')[0]
    scs.loc[scs['Alert'].str.contains(r'C[0-9]{4}PTT[0-9]{3}'), 'Asset Code'] = (
        scs.loc[scs['Alert']
               .str.contains(r'C[0-9]{4}PTT[0-9]{3}')]['Alert']
        .str.extract('(C[0-9]{4}PTT[0-9]{3})')[0].str.replace('02', ''))
    # Label destackers/stackers as PLC    
    scs.loc[scs['PLCN'] > 34,
                 'Asset Code'] = scs.loc[scs['PLCN'] > 34,
                 'PLC']    
    # Left over ones label with same code as their PLC label
    scs.loc[(scs['Asset Code'].isna()) & (scs['Desk'] == 'Z'), 'Asset Code'] = scs.loc[(scs['Asset Code'].isna()) & (scs['Desk']=='Z'), 'Alert'].apply(lambda x:x.split(':')[1].strip())
    scs['Asset Code'].fillna('Unable to extract')
    return scs


@logger.logger
def add_tote_colour(scs_code):
    """
    Summary
    -------
    Assings a tote colour, area and PLC Number to each fault and
    identifies any pick stations
    Parameters
    ----------
    scs_code: pandas DataFrame
        dataframe of faults
    asset_lu: pandas DataFrame
        asset lookup
    Returns
    -------
    df_totes: pandas DataFrame
        dataframe with 'Tote Colour','Area','Pick Station' and 'PLCN'
    unmapped: pandas DataFrmae
        dataframe with the unmapped asset code and a count of their occurance
    Example
    --------
    df, unmapped = add_tote_colour(data)
    """
    # Loads asset look up table
    asset_lu = load_tote_lookup()
    # Merges look up to label with tote colour and area
    df_totes = pd.merge(scs_code, asset_lu.drop('Number', axis=1), how='left',
                        on='Asset Code')
    # Sets external PLCs as Blue
    df_totes.loc[df_totes['PLC'].isin(['C17', 'C16', 'C15', 'C23']),
                 'Tote Colour'] = 'Blue'
    # Labeling pick stations and assigning colour to both
    df_totes['Pick Station'] = df_totes['Alert'].str.extract(
        r'(PTT[0-9]{3})').fillna(False)
    df_totes.loc[(df_totes['Pick Station'] != False),
                 'Tote Colour'] = 'Both'
    # Label destacker/stacker faults as blue
    df_totes.loc[df_totes['PLCN'] > 34,
                 'Tote Colour'] = 'Blue'
    # Label C15,16,23,stacker and pickstation fault areas
    df_totes.loc[df_totes['PLC'].isin(['C16', 'C15', 'C23']), 'Area'] = (
        df_totes.loc[df_totes['PLC'].isin(['C16', 'C15', 'C23']), 'PLC'])
    df_totes.loc[df_totes['Pick Station'] != False, 'Area'] = 'PTT'
    df_totes.loc[df_totes['PLCN'] > 34, 'Area'] = 'Stacker/Destacker'
    df_totes.loc[df_totes['Area'].isnull() * df_totes['Desk'] == 'Z',
                 'Area'] = 'PLC External'
    # Label any faults which haven't been mapped to an area
    df_totes['Area'] = df_totes['Area'].fillna('Unknown')

    # Create a dataframe of unmapped assets
    unmapped = df_totes[df_totes['Area'] == 'Unknown']['Asset Code'].value_counts().reset_index().copy()
    unmapped = unmapped.rename(columns={'index': 'Asset', 'Asset Code': 'Occurrence'})
    # Map PLC external to both and label unknowns
    df_totes.loc[df_totes['Area'] == 'PLC External', 'Tote Colour'] = 'Both'
    df_totes['Tote Colour'] = df_totes['Tote Colour'].fillna('Unknown')

    return df_totes, unmapped


@logger.logger
def pre_process_fa(fa, remove_same_location_faults=True, remove_warnings=True, remove_door=True):

    """
    Summary
    -------
    Takes raw fault dataframe, maps fault to areas, types and tote colours,
    removes warnings and duplicate faults and converts duration and timestamp
    to timedeltas and datetime objects
    Parameters
    ----------
    fa: pandas DataFrame
        raw fault dataframe
    remove_same_location_faults: Bool
        option to remove fault that happens in the same place at the same time
    remove_warning: Bool
        remove alerts idenfied by engineering as warnings rather than faults
    remove_door: Bool
        option to remove carousel door open faults over an hour
    Returns
    -------
    fa: pandas DataFrame
        processed fault dataframe
    unmapped: pandas DataFrmae
        dataframe with the unmapped asset code and a count of their occurance
    Example
    --------
    fa, unmapped = pre_process_fa(fa,
                                  remove_same_location_faults=True,
                                  remove_warnings=True,
                                  remove_door=True)
    """

    fa.columns = pd.Series(fa.columns).str.strip()
    # Add asset codes and tote colour to dataframe
    fa = add_code(fa)
    fa, unmapped = add_tote_colour(fa)

    fa.reset_index(inplace=True)

    fa.rename(columns={'Entry Time': 'timestamp', 'index': 'Alert ID'}, inplace=True)

    # Assign PLC code to Quadrants
    Quad_1 = ['C0' + str(i) for i in range(5, 8)] + \
        ['SCSM0' + str(i) for i in range(1, 6)]
    Quad_2 = ['C0' + str(i) for i in range(8, 10)] + \
        ['SCSM0' + str(i) for i in range(7, 10)] + ['SCSM10']
    Quad_3 = ['C' + str(i) for i in range(10, 13)] + \
        ['SCSM' + str(i) for i in range(11, 16)]
    Quad_4 = ['C' + str(i) for i in range(13, 15)] + \
        ['SCSM' + str(i) for i in range(17, 21)]

    # Assign faults to Quadrants
    Quad = []
    for i in fa['PLC']:
        if i in Quad_1:
            Quad.append(1)
        elif i in Quad_2:
            Quad.append(2)
        elif i in Quad_3:
            Quad.append(3)
        elif i in Quad_4:
            Quad.append(4)
        else:
            Quad.append(0)
    fa['Quadrant'] = Quad

    lu = load_module_lookup()
    # Copy desk
    fa['Desk_edit'] = fa['Desk']
    # Mark carousels
    fa.loc[fa['PLC'].str.contains(r'SCS', regex=True), 'Desk_edit'] = \
        fa.loc[fa['PLC'].str.contains(r'SCS', regex=True), 'PLC']
    # Mark PTTs
    fa.loc[~(fa['Pick Station'].isin([False])), 'Desk_edit'] = \
        fa[~(fa['Pick Station'].isin([False]))]['Pick Station'].apply(lambda x: x[:-1])
    # Set NA desk for outside assets
    fa.loc[fa['PLC'].isin(['C23', 'C16', 'C15', 'C17']), 'Desk_edit'] = 'X'
    fa.loc[fa['PLCN'] > 34, 'Desk_edit'] = 'X'
    # Merge module lookup
    fa = pd.merge(fa, lu, how='left', on=['PLC', 'Desk_edit']).drop('Desk_edit', axis=1)
    # Convert to datetime object
    fa['timestamp'] = pd.to_datetime(fa['timestamp'], dayfirst=True)
    # Label destacker alerts that are warnings
    fa['0 Merger'] = fa['Alert'].str.contains(r'extended|retracted')

    id_lu = load_ID_lookup()
    # Merge ID lookup to label alerts as warning or faults
    fa = fa.merge(id_lu, how='outer', on=["Fault ID", "0 Merger"])

    fa.drop('0 Merger', axis=1, inplace=True)

    # drop rows that engineering have identified as warnings
    if remove_warnings:
        fa = fa[fa['Alert Type'] != 'Warning']

    # drop rows where there is no duration data
    fa = fa.dropna(subset=['Duration'])

    # convert duration string to time delta and then to seconds (float)
    fa['Duration'] = pd.to_timedelta(fa['Duration'].str.slice(start=2))
    fa['Duration'] = fa['Duration'].dt.total_seconds()

    # Remove carousel door fault if over an hour
    if remove_door:
        fa = (fa[~((fa['Duration'] > 3600) &
              (fa['Alert'].str.contains('access door')))])

    # drop faults that happen at same time and in same location
    # (keep only the one with max duration)
    if remove_same_location_faults:
        fa = fa.sort_values('Duration').drop_duplicates(
            subset=['timestamp', 'PLC', 'Desk'],
            keep='last')

    # Remove module number for Quadrant loop faults
    fa.loc[fa['Loop'] == 'Quadrant', 'MODULE'] = np.nan

    # Drop ECB Faults
    fa = fa[~fa['PLC'].isin(['C17', 'SCSM22'])]

    return fa, unmapped


@logger.logger
def floor_shift_time_fa(fa, shift=0, duration_thres=0):
    """
    Summary
    -------
    Take process fault data, splits long faults into hours, floor entry time,
    marks with an end time, shift time desired amount, removes short faults if
    required and transforms duration to log relationship
    Parameters
    ----------
    fa: pandas DataFrame
        processed fault dataframe
    shift: Numeric
        shifts entry time desired amount to account for delay between faults
        occuring and their impact on availability
    duration_thres: Numberic
        only faults long that this will be kept
    Returns
    -------
    fa_floor: pandas DataFrame
        floored, shifted, filtered and transformed faults dataframe
    Example
    --------
    fa, unmapped = pre_process_fa(fa,
                                  remove_same_location_faults=True,
                                  remove_warnings=True,
                                  remove_door=True)
    """

    fa_floor = fa.copy()
    fa_floor['Entry Time'] = fa_floor['timestamp'].copy()
    # Remove rows shorter than duration threshold
    fa_floor = fa_floor[fa_floor['Duration'] > duration_thres]
    # Shifts entry time by desired amount
    fa_floor['timestamp'] = fa_floor['timestamp'].apply(
        lambda x: x + pd.to_timedelta(shift, unit='m'))
    fa_floor.sort_values('Duration', ascending=False, inplace=True)

    # Start and end of entry hour labeled
    fa_floor['Start'] = fa_floor['timestamp'].dt.floor('H')
    fa_floor['End'] = fa_floor['Start'].apply(
        lambda x: x + pd.to_timedelta(1, unit='h'))
    # Time passed in hour before fault started and time left in hour labeled
    fa_floor['Time Passed'] = pd.to_timedelta(
        fa_floor['timestamp'] - fa_floor['Start']).dt.total_seconds()
    fa_floor['Time Left'] = pd.to_timedelta(
        fa_floor['End'] - fa_floor['timestamp']).dt.total_seconds()

    # The number of hours the faults spans labeled
    fa_floor['Hours'] = np.ceil(
        (fa_floor['Duration'] + fa_floor['Time Passed']) / 3600)
    # Duplicate faults which span multiple hours
    fa_floor = fa_floor.loc[fa_floor.index.repeat(
        fa_floor.Hours)].reset_index(drop=True)
    # Label which number duplicate each duplicate row is
    fa_floor['Counts'] = fa_floor.groupby(['Alert ID',
                                           'Hours',
                                           'timestamp']).cumcount()
    # Add the number of hours from the counts column to start time as the new timestamp
    fa_floor['timestamp'] = fa_floor['Start'] + \
        pd.to_timedelta(fa_floor['Counts'], unit='h')
    # Add number of hour in counts column to entry time
    fa_floor['Entry Time'] = fa_floor['Entry Time'] + \
        pd.to_timedelta(fa_floor['Counts'], unit='h')
    fa_floor.reset_index(inplace=True, drop=True)

    for i in fa_floor.index:
        if fa_floor['Counts'][i] > 0:
            # If the row is a repeat, floor the entry time
            fa_floor.loc[i, 'Entry Time'] = fa_floor.loc[i, 'Entry Time'].floor('H')

        if fa_floor['Counts'][i] == 0 and (fa_floor['Duration'][i] +
                                           fa_floor['Time Passed'][i]) > 3600:
            # If row is the first repeated row
            # set the duration equal to time left in the hour
            fa_floor.loc[i, 'Duration'] = fa_floor.loc[i, 'Time Left']
            # Take away the duration allocated to this hour from the next hour
            fa_floor.loc[i + 1, 'Duration'] = fa_floor.loc[i + 1, 'Duration'] - \
                fa_floor.loc[i, 'Duration']
        elif fa_floor['Counts'][i] != 0 and fa_floor['Duration'][i] > 3600:
            # If the row is a repeated row and there is more than an hour of duration
            # Then take an hour off the next row's duration
            fa_floor.loc[i + 1, 'Duration'] = fa_floor.loc[i, 'Duration'] - 3600
            # Set this rows duration equal to an hour
            fa_floor.loc[i, 'Duration'] = 3600

    # End Time set to entry time plus duration
    fa_floor['End Time'] = fa_floor['Entry Time'] + \
        fa_floor['Duration'].apply(lambda x: pd.to_timedelta(x, unit='s'))

    fa_floor.sort_values('timestamp', ascending=True, inplace=True)
    # Drop columns not required anymore
    fa_floor.drop(['Start', 'End', 'Time Passed',
                   'Time Left', 'Hours', 'Counts'], axis=1, inplace=True)
    fa_floor.reset_index(inplace=True, drop=True)

    # Transform duration to log of duration + 1
    fa_floor['Duration'] = np.log(fa_floor['Duration']) + 1

    return fa_floor


@logger.logger
def fault_select(data, modules, PTT='None'):
    """
    Summary
    -------
    Selects the follow faults
    Anything in same Module
    PLC External applying to that PLC
    Quadrant loop dataults for the module that quadrant is in
    Outer
    and returns reduce faults dataframe
    Parameters
    ----------
    data: pandas DataFrame
        dataframe of faults
    module: numeric
        modules to select
    PTT: string
        pick station to select within module
    Returns
    -------
    faults_mod: pandas DataFrame
        dataframe of selected faults
    Example
    --------
    fa = fault_select(data, modules=[1], PTT='PTT011')
    """
    data = data.copy()
    ptt_du = load_PTT_lookup()

    # Drop Old Pick Station Column
    data.drop('Pick Station', axis=1, inplace=True)
    # Merge to create new pick station column labeling pick stations and pick
    # station specific assets
    data = data.merge(ptt_du, how='outer', on='Asset Code')
    data['Pick Station'] = data['Pick Station'].fillna(False)
    # Select faults in same module
    mod_str = pd.Series(modules).astype('str')
    faults1 = data[data['MODULE'].isin(mod_str)]
    # Select PLC external faults in the same PLC as module
    a = data[['PLC', 'MODULE']].drop_duplicates()
    b = a[(a['MODULE'].isin(mod_str)) & a['PLC'].str.contains('^C', regex=True)]
    c = b['PLC'] + ' External'
    faults2 = data[data['MODULE'].isin(list(c))]
    # Select quadrant loop faults from same quadrant
    q = data[data['MODULE'].isin(mod_str)]['Quadrant']
    faults3 = data[(data['Loop'].isin(['Quadrant'])) & data['Quadrant'].isin(q)]
    # Select all outside quadrant faults
    faults4 = data[data['Loop'].isin(['Outside'])]

    # Merge 4 selected dataframes and drop duplicates
    faults_mod = pd.concat([faults1, faults2, faults3, faults4])
    faults_mod.drop_duplicates(inplace=True)

    # Keep only non-pick station specifc faults and faults specific to selected
    # pick station
    if PTT != 'None':
        faults_mod = faults_mod[faults_mod['Pick Station'].isin([PTT, False])]

    return faults_mod


@logger.logger
def faults_aggregate(df, fault_agg_level, agg_type='sum'):
    """
    Summary
    -------
    Aggregates faults up into hours and selected aggregation group

    Parameters
    ----------
    df: pandas DataFrame
        faults dataframe
    fault_agg_level: string
        required grouping of faults
    agg_type: string
        how to aggregate faults
    Returns
    -------
    df: pandas DataFrame
        aggregated faults dataframe
    Example
    --------
    fa = faults_aggregate(df,
                          fault_agg_level = 'Asset',
                          agg_type='sum')
    """
    df = df.copy()

    if fault_agg_level == 'None':
        # If no agg_level set, then just group all faults by hour and aggregate
        df = df.groupby('timestamp', as_index=False).agg({'Duration': agg_type})
        df = df.set_index('timestamp')
    else:
        # Group by agg level and hour, aggreate duration and pivot on agg_type
        df = df.groupby(['timestamp', fault_agg_level],
                        as_index=False).agg({'Duration': agg_type})
        df = pd.pivot_table(df, values='Duration', index='timestamp',
                            columns=fault_agg_level, fill_value=0)

    return df


@logger.logger
def av_at_select(av, at, availability_select_options="None",
                 remove_high_AT=True, AT_limit="None"):
    """
    Summary
    -------
    Selects availability and active tote dataframes based
    on option arguments and remove active tote high values
    ----------
    av: pandas DataFrame
        processed availability dataframe
    at: pandas DataFrame
        processed active tote dataframe
    availability_select_options: dictionary
        column to select on and values to select
    remove_high_AT: Bool
        option to remove active tote outliers
    AT_limit: Numeric
        set active totes above limit to limit
    Returns
    -------
    av: pandas DataFrame
        selected availability dataframe
    at: pandas DataFrame
        selected active tote dataframe
    Example
    --------
    av, at=av_at_select(av,
                        at,
                        availability_select_options={'Module': [1]},
                        remove_high_AT=True,
                        AT_limit="None")
    """
    av = av.copy()
    at = at.copy()

    if availability_select_options != "None":
        for i in availability_select_options.keys():
            # Select av rows based on options
            av = av[av[i].isin(availability_select_options[i])]

            if i == 'Pick Station':
                # If select option is pickstation, extract module
                mod_str = [w[3:5] for w in availability_select_options[i]]
                mod = [int(i) for i in mod_str]
                # The select active totes rows in that module
                at = at[at['Module'].isin(mod)]

            elif i == 'Module' or i == 'Quadrant':
                # Select row in specified module or quadrant
                at = at[at[i].isin(availability_select_options[i])]
            else:
                print(r'\nNot a valid level, returned all data\n')

    if remove_high_AT:
        # Create pivot table of totes and module
        at_piv = pd.pivot_table(at, values='TOTES', index='timestamp',
                                columns='Module')
        # Create AT look table by take mean active totes per module
        at_lookup = at.groupby('Module').mean().drop('Quadrant', axis=1)

        # Calculate IQR and set outlier limt for each module
        Limit = []
        for i in at_piv.columns:
            Q1 = at_piv[i].quantile(0.25)
            Q3 = at_piv[i].quantile(0.75)
            Limit.append(Q3 + 1.5 * (Q3-Q1))
        # Append lookup with upper limit
        at_lookup['Upper limit'] = Limit
        at_lookup.drop('TOTES', axis=1, inplace=True)
        # Merge lookup then remove row above upper limit
        at = at.join(at_lookup, how='inner', on='Module')
        at = at[at['TOTES'] <= at['Upper limit']]
        at.drop('Upper limit', axis=1, inplace=True)

    if AT_limit != "None":
        # Set any values above limit to the limit
        at['TOTES'] = at['TOTES'].clip(0, AT_limit)

    return av, at


@logger.logger
def aggregate_availability(df, agg_level='None'):
    """
    Summary
    -------
    Aggregates availability up into hours
    and selected aggregation group
    Parameters
    ----------
    df: pandas DataFrame
        availability dataframe
    agg_level: string
        required grouping of availability
    Returns
    -------
    df: pandas DataFrame
        aggregated availability dataframe
    Example
    --------
    av = aggregate_availability(df,
                          fault_agg_level = 'Asset')
    """
    df = df.copy()

    if agg_level == 'None':
        # If no agg level, groupby hour and take mean availability
        df = df.groupby(['timestamp'],
                        as_index=False).agg({'Availability': 'mean'})
    else:
        # groupby hour and agg_level and take mean availability
        df = df.groupby(['timestamp',
                        agg_level], as_index=False).agg({'Availability': 'mean'})

    df = df.set_index('timestamp')

    return df


@logger.logger
def aggregate_totes(active_totes, agg_level='None'):
    """
    Summary
    -------
    Aggregates active totes up into hours and
    selected aggregation group
    Parameters
    ----------
    active_totes: pandas DataFrame
        active totes dataframe
    agg_level: string
        required grouping of active totes
    Returns
    -------
    active_totes: pandas DataFrame
        aggregated active totes dataframe
    Example
    --------
    at = aggregate_totes(at,
                          fault_agg_level = 'Asset')
    """
    active_totes = active_totes.copy()
    # Floor timestamp to allow groupby hours
    active_totes['timestamp'] = active_totes['timestamp'].dt.floor('H')

    if agg_level == 'Module' or agg_level == 'PTT':
        # Group active totes by module and hour and take mean
        active_totes = active_totes.groupby(['timestamp', 'Module'],
                                            as_index=False).mean()
        active_totes.drop('Quadrant', axis=1, inplace=True)
    elif agg_level == 'Quadrant':
        # Group active totes by quadrant and hour and take mean
        active_totes = active_totes.groupby(['timestamp', 'Quadrant'],
                                            as_index=False).mean()
        active_totes.drop('Module', axis=1, inplace=True)
    else:
        # Group active totes by hour and take mean
        active_totes = active_totes.groupby('timestamp', as_index=False).mean()
        active_totes.drop(['Module', 'Quadrant'], axis=1, inplace=True)

    active_totes = active_totes.set_index('timestamp')

    return active_totes


@logger.logger
def weight_hours(df, weights=[1]):
    """
    Summary
    -------
    adds weighted durations from previous hours to duration
    of the current hour
    Parameters
    ----------
    df: pandas DataFrame
        aggregated faults dataframe
    weights: list
        weighting for hours ([current,1 hour ago, 2 hours ago etc.])
    Returns
    -------
    df_weight: pandas DataFrame
        weighted fault dataframe
    Example
    --------
    fa_weight = weight_hours(fa,
                             weights=[0.7, 0.2, 0.1])
    """
    # Create dataframe of zeros same size as faults
    df_weight = pd.DataFrame(data=np.zeros(df.shape),
                             index=df.index, columns=df.columns)

    for i in range(len(df)):
        # iterate through each row in orginal df required in row of new df
        for x in range(len(weights)):
            if i-x >= 0:
                # Add the weighted row from df to the df_weight dataframe
                df_weight.iloc[i] = df_weight.iloc[i] + df.iloc[i-x]*weights[x]

    return df_weight


@logger.logger
def merge_av_fa_at(av_df, fa_df, at_df, min_date=None, max_date=None,
                   agg_level='None'):
    """
    Summary
    -------
    marges all 3 dataframes in one dataset
    Parameters
    ----------
    av_df: pandas DataFrame
        aggregated availability dataframe
    fa_df: pandas DataFrame
        aggregated faults dataframe
    at_df: pandas DataFrame
        aggregated active tote dataframe
    min_date: string
        optional lower date limit
    max_date: string
        optional upper date limit
    agg_level: string
        column to merge on
    Returns
    -------
    df: pandas DataFrame
        merged dataframe
    Example
    --------
    df = merge_av_fa_at(av_df,
                        fa_df,
                        at_df,
                        min_date=None,
                        max_date=None,
                        agg_level='Asset Code')
    """
    av_df = av_df.copy()
    fa_df = fa_df.copy()
    at_df = at_df.copy()

    if min_date is not None:
        # Minimum date is maximum of the minimum dates of the 3 dataframes
        # and the min_date
        min_date = max(av_df.index.min(), fa_df.index.min(), at_df.index.min(),
                       min_date)
    else:
        # Minimum date is maximum of the minimum dates of the 3 dataframes
        min_date = max(av_df.index.min(), fa_df.index.min(), at_df.index.min())
    if max_date is not None:
        # Maximum date is minimum of the maximum dates of the 3 dataframes
        # and the max_date
        max_date = min(av_df.index.max(), fa_df.index.max(), at_df.index.max(),
                       max_date)
    else:
        # Maximum date is minimum of the maximum dates of the 3 dataframes
        max_date = min(av_df.index.max(), fa_df.index.max(), at_df.index.max())

    # Select row inbetween min and date dates
    fa_df = fa_df.loc[min_date:max_date]
    av_df = av_df.loc[min_date:max_date]
    at_df = at_df.loc[min_date:max_date]

    if agg_level == 'None':
        # Only select availablity column for merge
        av_df = av_df["Availability"]
        # Inner merge av and fa on date index
        df = av_df.merge(fa_df, how='inner', left_on=None, right_on=None,
                         left_index=True, right_index=True)
        # Inner df and at on date index
        df = df.merge(at_df, how='inner', left_on=None, right_on=None,
                      left_index=True, right_index=True)
        df.reset_index(inplace=True)
    else:
        # Only select availablity and merge column for merge
        av_df = av_df[["Availability", agg_level]]
        av_df.reset_index(inplace=True)
        at_df.reset_index(inplace=True)
        fa_df.reset_index(inplace=True)

        # Inner merge av and fa on date index and agg_level
        df = av_df.merge(fa_df, how='inner', on='timestamp')
        # Inner merge df and at on date index and agg_level
        df = df.merge(at_df, how='inner', on=['timestamp', agg_level])
        df.drop([agg_level], axis=1, inplace=True)

    return df


@logger.logger
def create_PTT_df(fa_floor, at, av, weights=None):
    """
    Summary
    -------
    Creates merged df for each pick station,containing
    faults that impact the pick station, that pick stations
    availability and the active totes for the module that
    pick station is in. It then joins all the data frames
    together
    The faults dataframe for each pick station are saved
    in a dictionary for use in reporting
    Parameters
    ----------
    fa_floor: pandas DataFrame
        aggregated availability dataframe
    at: pandas DataFrame
        aggregated faults dataframe
    av: pandas DataFrame
        aggregated active tote dataframe
    weights: list
        weighting for hours ([current,1 hour ago, 2 hours ago etc.])
    Returns
    -------
    df_PTT: pandas DataFrame
        all the PTT dataframes joined together into one feature set
    fa_PTT: dictionary
        contains all the fault dataframes for each pick station
    Example
    --------
    df_PTT, fa_PTT=create_PTT_df(fa_floor,
                                 at,
                                 av,
                                 weights=[0.7, 0.3])
    """
    # Create a list of the pick stations in the data
    pick_stations = av['Pick Station'].unique()

    # Set up empty list and dataframes to be filled
    df_PTT = pd.DataFrame()
    fa_PTT = []
    for PTT in pick_stations:
        # Extract pick station module
        mod = int(PTT[3:5])
        fa_floor = fa_floor.copy()
        # Select faults that impact the pickstation
        fa_sel = fault_select(fa_floor, modules=[mod], PTT=PTT)
        # Agggregate the faults on asset codes
        fa_agg = faults_aggregate(fa_sel, fault_agg_level='Asset Code')
        if weights is not None and weights != [1]:
            # weight hours if option is selected
            fa_agg = weight_hours(df=fa_agg, weights=weights)

        # Select av and at data relevant to pick station and remove at outliers
        av_sel, at_sel = (
            av_at_select(av, at, availability_select_options={'Pick Station': [PTT]},
                         remove_high_AT=True, AT_limit='None'))

        # Aggregate av and at by module
        agg_level = 'Module'
        av_agg = aggregate_availability(av_sel, agg_level=agg_level)
        at_agg = aggregate_totes(at_sel, agg_level=agg_level)

        # Merge 3 dataframes on module
        df = merge_av_fa_at(av_df=av_agg, at_df=at_agg, fa_df=fa_agg,
                            agg_level=agg_level)

        # Append dataframe to df_PTT
        df_PTT = pd.concat([df_PTT, df], axis=0, join='outer', sort=False)
        # Append fault dataframe to fa_PTT
        fa_sel = fa_floor['Asset Code'].unique()
        fa_PTT.append(fa_sel)

    # Create fa_PTT dictionary with pick station as keys and fa dataframes as values
    fa_PTT = dict(zip(pick_stations, fa_PTT))

    df_PTT = df_PTT.fillna(0)

    # Move TOTES column to the end
    totes_col = df_PTT.pop('TOTES')
    df_PTT['TOTES'] = totes_col

    return df_PTT, fa_PTT


def log_totes(df):
    """
    Summary
    -------
    Takes features DataFrame, removes low TOTES, takes the natural log and
    drops the original TOTES column.
    ----------
    df: pandas DataFrame
        dataframe of features
    Returns
    -------
    df: pandas DataFrame
        dataframe with 'log_totes' column
    Example
    --------
    df_log = log_totes(data)
    """
    df = df.copy()
    # Remove totes row less than 5 as they negatively impact model
    df = df[df['TOTES'] > 5]
    # Transform tote to log and drop totes
    df['log_totes'] = np.log(df['TOTES'])
    df = df.drop(['TOTES'], axis=1)

    # Drop any infinite values
    df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    return df
