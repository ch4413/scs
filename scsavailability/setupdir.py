# Import dependencies
import pandas as pd
import os
import sys
from pathlib import Path


def create_log(path):
    """
    Summary
    -------
    creates empty run log and saves to directory
    Parameters
    ----------
    path: str
        save path
    Returns
    -------
    Example
    --------
    create_log(package_path)
    """
    # Create empty log dataframe
    log = pd.DataFrame(columns=['Run_ID',
                                'Timestamp',
                                'R2',
                                'Selected_Features',
                                'Time',
                                'Train_Data_Start',
                                'Train_Data_End',
                                'Export_Time'])
    try:
        # Write as csv to directory
        log.to_csv(path, index=False)
    except Exception:
        # If this fails, exit and send error message to terminal
        sys.exit('Failed to Create Log, check package path in config')


def create_cache(path):
    """
    Summary
    -------
    creates simple cache and saves to directory
    Parameters
    ----------
    path: str
        save path
    Returns
    -------
    Example
    --------
    create_cache(package_path)
    """
    # Create cache dataframe
    cache = pd.DataFrame({'Entry Time': {1: '01/01/2020 00:00:00'}})
    try:
        # Write cache to directory
        cache.to_csv(path, index=False)
    except Exception:
        # If this fails, exit and send error message to terminal
        sys.exit('Failed to Create Cache, check package path in config')


def setupdirectory(package_path):
    """
    Summary
    -------
    Checks cache, log and output folder to see if
    they exist and are in the right format, creates
    or replaces if they don't exist or are in the wrong
    format ensuring the directory is set up correctly
    before running the model
    Parameters
    ----------
    path: str
        directory path
    Returns
    -------
    Example
    --------
    setupdirectory(package_path)
    """
    # Create path strings
    cache_path = r'%scache.csv' % package_path
    log_path = r'%srun_log.csv' % package_path
    output_path = r'%soutputs' % package_path
    archive_path = r'%soutputs\Archive' % package_path

    # Check if log exists
    checklog = Path(log_path)
    if checklog.is_file():
        # If the path exists and is a file
        # Read log and assert it has 8 columns
        try:
            log = pd.read_csv(log_path)
            if log.shape[1] == 8:
                pass
            else:
                # If not, create new blank log
                create_log(log_path)
        except Exception:
            # If this fails to load, create new log
            create_log(log_path)
    else:
        # If log doesn't exist, create new log
        create_log(log_path)

    # Check if cache exists
    checkcache = Path(cache_path)
    if checkcache.is_file():
        # If the path exists and is a file
        # Read log and assert it has a entry time column
        # and that the column isn't empty
        try:
            cache = pd.read_csv(log_path)
            if cache['Entry Time'].empty:
                # If empty column, create new cache
                create_cache(cache_path)
            else:
                pass
        except Exception:
            # If this fails to load, create new cache
            create_cache(cache_path)
    else:
        # If the file doesn't exist, create new cache
        create_cache(cache_path)

    # Check if output directory exists
    checkoutput = Path(output_path)
    if checkoutput.is_dir():
        pass
    else:
        try:
            # If it doesn't exist, create it
            os.mkdir(output_path)
        except Exception:
            # If it fails to create exit code with message
            sys.exit("Creation of the directory %s failed, check package path in config" % output_path)

    # Check if archive directory exists
    checkarchive = Path(archive_path)
    if checkarchive.is_dir():
        pass
    else:
        try:
            # If it doesn't exist, create it
            os.mkdir(archive_path)
        except Exception:
            # If it fails to create exit code with message
            sys.exit("Creation of the directory %s failed, check package path in config" % archive_path)
