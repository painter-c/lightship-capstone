import transforms as T
import numpy as np
import pandas as pd


dummy_data = {'date_a': [np.datetime64('2000-01-01T12:00'),
                         np.datetime64('2000-01-01T12:00')],
              'date_b': [np.datetime64('2000-01-02T13:00'),
                         np.datetime64('2000-01-02T13:00')],
              'ignore': 'ignore'}

dummy_df = pd.DataFrame(dummy_data)


def test_dayofweek_transform(df):
    result = T.dayofweek_transform(df)
    print(result)
    
def test_timeofday_transform(df):
    result = T.timeofday_transform(df)
    print(result)