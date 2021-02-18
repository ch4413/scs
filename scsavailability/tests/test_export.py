from pytest import mark
import pandas as pd
import numpy as np
import pkg_resources
import scsavailability
import os

@mark.pipeline
@mark.run
class ExportTests:
    def test_export(self):
        output = pd.read_csv("./scsavailability/tests/TestData/ML_Test_Output.csv")
        output.to_csv("./outputs/ML_Output_999999.csv")
        log = pd.read_csv("./run_log.csv")
        new_row = pd.DataFrame([[999999, 'Exp Test', 'Exp Test', 'Exp Test', 'Exp Test', 'Exp Test', 'Exp Test','Not exported yet']],
                               columns=log.columns)
        new_log = log.append(new_row, ignore_index=True)
        new_log.to_csv("./run_log.csv", index=False)
        os.system("python -m scsavailability.export --config=./scsavailability/tests/TestData/test_config.yml")
        log = pd.read_csv("./run_log.csv")
        value = log.iloc[-1,7]
        log.drop(log.tail(1).index,inplace=True)
        log.to_csv("./run_log.csv", index=False)
        os.remove("./outputs/Archive/ML_Output_999999.csv")
        assert value != 'Not Exported Yet'



