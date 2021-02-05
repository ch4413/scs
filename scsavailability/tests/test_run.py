from pytest import mark
import pandas as pd
import numpy as np
import pkg_resources
import scsavailability
import os

@mark.pipeline
@mark.run
class RunTests:
    def test_run(self):
        os.system("python -m scsavailability.run --config=./scsavailability/tests/TestData/test_config.yml")
        log = pd.read_csv("./run_log.csv")
        value = log.iloc[-1,2]
        assert value == 'Test'



