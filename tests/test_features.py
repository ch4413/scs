from pytest import mark
from scsavailability import features as feat
import pandas as pd

@mark.feature
class FeatureTests:
    def test_load_module_lookup(self):
        lu = feat.load_module_lookup()
        assert list(lu.columns) == ['PLC','Desk_edit','MODULE']

    def test_load_tote_lookup(self):
        lu = feat.load_tote_lookup()
        assert list(lu.columns) == ['Area','Tote Colour','Loop','Suffix','Number','Asset Code']

    def test_load_ID_lookup(self):
        lu = feat.load_ID_lookup()
        assert list(lu.columns) == ['Fault ID','0 Merger','Alert Type']

    def test_load_PTT_lookup(self):
        lu = feat.load_PTT_lookup()
        assert list(lu.columns) == ['Asset Code','Pick Station']

    def test_preprocess_AT(self):
        at = pd.DataFrame({'ID':[1,5,7,11,20],'MODULE_ASSIGNED':['SCS01','SCS05','SCS07','SCS11','SCS20'],'TOTES':['42','30','21','55','90'],'DAY':['9','10','11','12','13'],'MONTH':['01','01','01','01','01'],'YEAR':['2021','2021','2021','2021','2021'],'HOUR':['08','08','08','08','08'],'MINUTE':['30','30','30','30','30']})
        at_processed = feat.pre_process_AT(at)
        at_expected = pd.DataFrame({'Module':[1,5,7,11,20],'TOTES':['42','30','21','55','90'],'timestamp':pd.to_datetime(['2021-01-09 08:30:00','2021-01-10 08:30:00','2021-01-11 08:30:00','2021-01-12 08:30:00','2021-01-13 08:30:00']),'Quadrant':[1,1,2,3,4]})
        assert at_processed.equals(at_expected)

    def test_preprocess_av(self):
        av = pd.DataFrame({'timestamp':['2021-01-09 08:00:00','2021-01-10 08:00:00','2021-01-11 08:00:00','2021-01-12 08:00:00','2021-01-13 08:00:00'],'Pick Station':['PTT011','PTT051','PTT072','PTT112','PTT202'],'Availability':[0.9,0.8,0.75,0.3,0.1]})
        av_processed = feat.pre_process_av(av)
        assert True

