from pytest import mark
from scsavailability import parser as ps
import pandas as pd
import numpy as np
import pkg_resources

@mark.pipeline
@mark.parser
class ModelTests:
    
    def test_parse_config(self):
        config = ps.parse_config(path = r'./scsavailability/tests/TestData/test_config.yml')

        assert config.path.source == 'Test'

