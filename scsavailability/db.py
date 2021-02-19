import pandas as pd
import urllib
from sqlalchemy import create_engine
import pkg_resources
from . import logger


@logger.logger
def get_credentials():
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
    filepath = pkg_resources.resource_filename(__name__, 'data/sql/sql_credentials.txt')
    with open(filepath, 'r') as f:
        user, password = f.readlines()[0].split()
    return user, password


@logger.logger
def mi_db_connection():
    """
    Summary
    -------
    Create connection to M&S MI database on Server
    Parameters
    ----------
    Returns
    -------
    sql_conn: SQLAlchemy Connection
        SQL connection
    Example
    --------
    sql_conn = mi_db_connection()
    """
    user, password = get_credentials()
    params = urllib.parse.quote_plus('Driver={SQL Server};'
                                     'Server=mshsrmnsukc0139;'
                                     'Database=ODS;'
                                     'UID=%s;'
                                     'PWD=%s' %(user,password))
                                                 
    sql_conn = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)                        
    return sql_conn


@logger.logger
def read_query(sql_conn, query_path='../sql/test_query.sql'):
    """
    Summary
    -------
    Read .sql file and return data from SQL connection
    ----------
    sql_conn: SQLAlchemy Connection
         SQL connection
    query_path: str
         path to .sql file
    Returns
    -------
    df: pandas DataFrame
        dataframe returned from query
    Example
    --------
    sql_conn = db.mi_db_connection()
    df = read_query(sql_conn,path)
    """
    fd = open(query_path, 'r')
    query = fd.read()
    fd.close()

    df = pd.read_sql(query, sql_conn)

    return df


@logger.logger
def output_to_sql(output, sql_conn, tablename='newton_AzurePrep_MLCoefficients',schema='SOLAR'):
    """
    Summary
    -------
    Writes pandas dataframe to SQL Server
    ----------
    output: Pandas DataFrame
        DataFrame to write to SQL server
    sql_conn: SQLAlchemy Connection
        SQL connection
    tablename: string
        table name in SQL Server to write to
    schema: string
        schema where table is located        
    Returns
    -------
    Example
    --------
    sql_conn = db.mi_db_connection()
    output_to_sql(output,
                  sql_conn,
                  tablename,
                  schema)
    """
    output.to_sql(name=tablename,
                  con=sql_conn,
                  schema=schema,
                  if_exists='replace',
                  index=False)
    print('Output sucessfully written to SQL Server')              
