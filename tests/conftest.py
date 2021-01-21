from pytest import fixture
import pandas as pd

@fixture(scope = 'function')
def load_test_data(file_name):
    data_path = "./TestData/"
    data = pd.read_csv(data_path + file_name)
    return data


