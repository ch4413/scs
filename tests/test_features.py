from pytest import mark
from scsavailability import features as ft
import pandas as pd

@mark.feature
class FeatureTests:
    def test_load_module_lookup(self):
        lu = ft.load_module_lookup()
        assert isinstance(lu, pd.DataFrame)
