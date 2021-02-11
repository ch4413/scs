from pytest import mark
from scsavailability import model as md
import pandas as pd
import numpy as np
import pkg_resources

@mark.package
@mark.model
class ModelTests:
    
    def test_gen_feat_var(self):
        df = pd.DataFrame({'timestamp': {0: pd.to_datetime('2021-01-09 08:00:00'),
                                        1: pd.to_datetime('2021-01-10 08:00:00'),
                                        2: pd.to_datetime('2021-01-11 08:00:00'),
                                        3: pd.to_datetime('2021-01-12 08:00:00'),
                                        4: pd.to_datetime('2021-01-13 08:00:00')},
                            'Availability': {0: 0.9, 1: 0.8, 2: 0.75, 3: 0.3, 4: 0.1},
                            'C05PTT011': {0: 6.480638923341991, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                            'C1502RDC070': {0: 7.835184586147302, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                            'C1603STA253': {0: 0.0, 1: 0.0, 2: 7.0867747269123065, 3: 0.0, 4: 0.0},
                            'C2301RDC176': {0: 0.0, 1: 8.512071245835466, 2: 0.0, 3: 0.0, 4: 0.0},
                            'C1002STA081': {0: 0.0, 1: 0.0, 2: 0.0, 3: 2.09861228866811, 4: 0.0},
                            'SCS116': {0: 0.0, 1: 0.0, 2: 0.0, 3: 4.401197381662156, 4: 0.0},
                            'SCSM20': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 8.922985958711195},
                            'log_totes': {0: 3.7376696182833684,
                                            1: 3.4011973816621555,
                                            2: 3.044522437723423,
                                            3: 4.007333185232471,
                                            4: 4.499809670330265}})

        X,y = md.gen_feat_var(df)

        X_expected = pd.DataFrame({'C05PTT011': {0: 6.480638923341991, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                                    'C1502RDC070': {0: 7.835184586147302, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                                    'C1603STA253': {0: 0.0, 1: 0.0, 2: 7.0867747269123065, 3: 0.0, 4: 0.0},
                                    'C2301RDC176': {0: 0.0, 1: 8.512071245835466, 2: 0.0, 3: 0.0, 4: 0.0},
                                    'C1002STA081': {0: 0.0, 1: 0.0, 2: 0.0, 3: 2.09861228866811, 4: 0.0},
                                    'SCS116': {0: 0.0, 1: 0.0, 2: 0.0, 3: 4.401197381662156, 4: 0.0},
                                    'SCSM20': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 8.922985958711195},
                                    'log_totes': {0: 3.7376696182833684,
                                                1: 3.4011973816621555,
                                                2: 3.044522437723423,
                                                3: 4.007333185232471,
                                                4: 4.499809670330265}})

        y_expected = pd.Series([0.9,0.8,0.75,0.3,0.1])

        assert X.equals(X_expected) and y.equals(y_expected)


    def test_split(self):

        X = pd.DataFrame({'C05PTT011': {0: 6.480638923341991, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                                    'C1502RDC070': {0: 7.835184586147302, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                                    'C1603STA253': {0: 0.0, 1: 0.0, 2: 7.0867747269123065, 3: 0.0, 4: 0.0},
                                    'C2301RDC176': {0: 0.0, 1: 8.512071245835466, 2: 0.0, 3: 0.0, 4: 0.0},
                                    'C1002STA081': {0: 0.0, 1: 0.0, 2: 0.0, 3: 2.09861228866811, 4: 0.0},
                                    'SCS116': {0: 0.0, 1: 0.0, 2: 0.0, 3: 4.401197381662156, 4: 0.0},
                                    'SCSM20': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 8.922985958711195},
                                    'log_totes': {0: 3.7376696182833684,
                                                1: 3.4011973816621555,
                                                2: 3.044522437723423,
                                                3: 4.007333185232471,
                                                4: 4.499809670330265}})

        y = pd.Series([0.9,0.8,0.75,0.3,0.1])

        X_train, X_test, y_train, y_test = md.split(X, y, split_options={'test_size': 0.3, 'random_state': 42})

        X_train_expected = pd.DataFrame({'C05PTT011': {2: 0.0, 0: 6.480638923341991, 3: 0.0},
                                        'C1502RDC070': {2: 0.0, 0: 7.835184586147302, 3: 0.0},
                                        'C1603STA253': {2: 7.0867747269123065, 0: 0.0, 3: 0.0},
                                        'C2301RDC176': {2: 0.0, 0: 0.0, 3: 0.0},
                                        'C1002STA081': {2: 0.0, 0: 0.0, 3: 2.09861228866811},
                                        'SCS116': {2: 0.0, 0: 0.0, 3: 4.401197381662156},
                                        'SCSM20': {2: 0.0, 0: 0.0, 3: 0.0},
                                        'log_totes': {2: 3.044522437723423,
                                                        0: 3.7376696182833684,
                                                        3: 4.007333185232471}})

        X_test_expected = pd.DataFrame({'C05PTT011': {1: 0.0, 4: 0.0},
                                        'C1502RDC070': {1: 0.0, 4: 0.0},
                                        'C1603STA253': {1: 0.0, 4: 0.0},
                                        'C2301RDC176': {1: 8.512071245835466, 4: 0.0},
                                        'C1002STA081': {1: 0.0, 4: 0.0},
                                        'SCS116': {1: 0.0, 4: 0.0},
                                        'SCSM20': {1: 0.0, 4: 8.922985958711195},
                                        'log_totes': {1: 3.4011973816621555, 4: 4.499809670330265}})

        y_train_expected = pd.Series({2: 0.75, 0: 0.9, 3: 0.3})

        y_test_expected = pd.Series({1: 0.8, 4: 0.1})

        assert X_train.equals(X_train_expected) and X_test.equals(X_test_expected) and y_train.equals(y_train_expected) and y_test.equals(y_test_expected)

    def test_run_OLS(self):

        stream = pkg_resources.resource_stream(__name__, '/TestData/ML_test_data.csv')
        data = pd.read_csv(stream)

        y = data['quality']
        X = data.drop('quality',axis=1)

        X_train, X_test, y_train, y_test = md.split(X, y, split_options={'test_size': 0.3, 'random_state': 42})

        r2_oos, coeff, num_assets = md.run_OLS(X_train=X_train, y_train=y_train,
                                               X_test=X_test, y_test=y_test, n=1)

        r2_oos = round(r2_oos,2)
        r2_oos_expected = 0.35
        coeff['Coefficient'] = coeff['Coefficient'].apply(lambda x:round(x,2))
        coeff_expected = pd.DataFrame({'Asset Code': {0: 'volatile acidity',
                                                    1: 'chlorides',
                                                    2: 'total sulfur dioxide',
                                                    3: 'pH'},
                                        'Coefficient': {0: -1.02,
                                                    1: -1.87,
                                                    2: 0.00,
                                                    3: -0.29}})
        num_assets_expected = 4

        assert r2_oos == r2_oos_expected and coeff.equals(coeff_expected) and num_assets == num_assets_expected








        

