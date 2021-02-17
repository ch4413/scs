from pytest import mark
from scsavailability import db
import pandas as pd

@mark.pipeline
@mark.db
# @mark.skip(reason='Not connected')
class DBTests:
    def test_read_query(self):
        conn = db.mi_db_connection()
        # Query DB using stored package queries
        at_path = r'./scsavailability/data/sql/active_totes.sql'
        av_path = r'./scsavailability/data/sql/availability.sql'
        fa_path = r'./scsavailability/data/sql/faults.sql'

        at = db.read_query(sql_conn=conn, query_path=at_path)
        av = db.read_query(sql_conn=conn, query_path=av_path)
        fa = db.read_query(sql_conn=conn, query_path=fa_path)

        at_cols = list(at.columns)
        av_cols = list(av.columns)
        fa_cols = list(fa.columns)

        at_cols_expected = ['ID', 'MODULE_ASSIGNED', 'TOTES', 'DAY', 
                            'MONTH', 'YEAR', 'HOUR', 'MINUTE']
        av_cols_expected = ['timestamp', 'Pick Station', 'Availability']
        fa_cols_expected = ['Number', 'Alert', 'Entry Time', 
                            'PLC', 'Desk', 'Duration', 'Fault ID']

        assert at_cols == at_cols_expected and av_cols == av_cols_expected and fa_cols == fa_cols_expected

 

