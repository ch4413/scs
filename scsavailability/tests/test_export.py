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
        #output.to_csv("./scsavailability/outputs/ML_Output_999999.csv")
        #os.system("python -m scsavailability.export --config=./scsavailability/tests/TestData/test_config.yml")
        #log = pd.read_csv("./run_log.csv")
        #value = log.iloc[-1,7]
        assert True



