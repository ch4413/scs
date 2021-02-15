import pandas as pd
import pyodbc


def get_credentials(filepath='../data/sql/sql_password.txt'):
    """
    Summary
    -------
    reads database credentials from .txt file and creates string objects of
    username and password
    Parameters
    ----------
    filepath: str
         filepath to .txt credentials file
    Returns
    -------
    user: str
        username string
    password: str
        password string
    Example
    --------
    user, password = get_credentials()
    """
    with open(filepath, 'r') as f:
        user, password = f.readlines()[0].split()
    return user, password


def mi_db_connection():
    """
    Summary
    -------
    Create connection to M&S MI database on Server without credentials.
    Parameters
    ----------
    Returns
    -------
    sql_conn: pyodbc Connection
        SQL connection
    Example
    --------
    sql_conn = db_connect('dbname', 'myuser', 'mypass')
    """
    sql_conn = pyodbc.connect('Driver={SQL Server};'
                              'Server=mshsrmnsukc0139;'
                              'Database=ODS;'
                              'as_dataframe=True;'
                              'UID=DS_NewtonSQLCATE;'
                              'PWD=Tu35day@01')
   # insert password as PWD and username as UID                           
    return sql_conn


def read_query(sql_conn, query_path='../sql/test_query.sql'):
    """
    Summary
    -------
    Read .sql file and return data from SQL connection
    ----------
    sql_conn: pyodbc Connection
         SQL connection
    query_path: str
         path to .sql file
    Returns
    -------
    df: pandas DataFrame
        dataframe returned from query
    Example
    --------
    sql_conn = db_connect('dbname', 'myuser', 'mypass')
    df = read_query(sql_conn)
    """
    fd = open(query_path, 'r')
    query = fd.read()
    fd.close()

    df = pd.read_sql(query, sql_conn)

    return df
