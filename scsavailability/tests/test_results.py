from pytest import mark
from scsavailability import results as rs
import pandas as pd
import numpy as np
import pkg_resources

@mark.package
@mark.results
class ResultsTests:
    
    def test_create_output(self):

        fa_PTT = {'PTT011': ['C1502RDC070', 'C05PTT011', 'C2301RDC176', 'C0702STA049',
                    'C1603STA253', 'SCS116', 'C1002STA081', 'SCSM20', 'C1402STA049'],
         'PTT051': ['C1502RDC070', 'C05PTT011', 'C2301RDC176', 'C0702STA049',
                    'C1603STA253', 'SCS116', 'C1002STA081', 'SCSM20', 'C1402STA049'],
         'PTT072': ['C1502RDC070', 'C05PTT011', 'C2301RDC176', 'C0702STA049',
                    'C1603STA253', 'SCS116', 'C1002STA081', 'SCSM20', 'C1402STA049'],
         'PTT112': ['C1502RDC070', 'C05PTT011', 'C2301RDC176', 'C0702STA049',
                    'C1603STA253', 'SCS116', 'C1002STA081', 'SCSM20', 'C1402STA049'],
         'PTT202': ['C1502RDC070', 'C05PTT011', 'C2301RDC176', 'C0702STA049',
                    'C1603STA253', 'SCS116', 'C1002STA081', 'SCSM20', 'C1402STA049']}
        
        coeff = pd.DataFrame({'Asset Code': {0: 'C1502RDC070',
                                            1: 'C2301RDC176',
                                            2: 'C1603STA253',
                                            3: 'SCSM20'},
                                'Coefficient': {0: -0.0001,
                                            1: -0.002,
                                            2: -0.04,
                                            3: -0.005}})

        Output = rs.create_output(fa_PTT, coeff)

        Output_expected = pd.DataFrame({'ID': {0: 'C1502RDC070',
                                                1: 'C2301RDC176',
                                                2: 'C1603STA253',
                                                3: 'SCSM20',
                                                4: 'C1502RDC070',
                                                5: 'C2301RDC176',
                                                6: 'C1603STA253',
                                                7: 'SCSM20',
                                                11: 'SCSM20',
                                                10: 'C1603STA253',
                                                9: 'C2301RDC176',
                                                8: 'C1502RDC070',
                                                12: 'C1502RDC070',
                                                13: 'C2301RDC176',
                                                14: 'C1603STA253',
                                                15: 'SCSM20',
                                                16: 'C1502RDC070',
                                                17: 'C2301RDC176',
                                                18: 'C1603STA253',
                                                19: 'SCSM20'},
                                        'COEFFICIENT': {0: -0.0001,
                                                        1: -0.002,
                                                        2: -0.04,
                                                        3: -0.005,
                                                        4: -0.0001,
                                                        5: -0.002,
                                                        6: -0.04,
                                                        7: -0.005,
                                                        11: -0.005,
                                                        10: -0.04,
                                                        9: -0.002,
                                                        8: -0.0001,
                                                        12: -0.0001,
                                                        13: -0.002,
                                                        14: -0.04,
                                                        15: -0.005,
                                                        16: -0.0001,
                                                        17: -0.002,
                                                        18: -0.04,
                                                        19: -0.005},
                                        'PTT': {0: 'PTT011',
                                                1: 'PTT011',
                                                2: 'PTT011',
                                                3: 'PTT011',
                                                4: 'PTT051',
                                                5: 'PTT051',
                                                6: 'PTT051',
                                                7: 'PTT051',
                                                11: 'PTT072',
                                                10: 'PTT072',
                                                9: 'PTT072',
                                                8: 'PTT072',
                                                12: 'PTT112',
                                                13: 'PTT112',
                                                14: 'PTT112',
                                                15: 'PTT112',
                                                16: 'PTT202',
                                                17: 'PTT202',
                                                18: 'PTT202',
                                                19: 'PTT202'}})

        assert Output.equals(Output_expected)