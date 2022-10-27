import transforms as T
import numpy as np
import pandas as pd
from utils import unhash_data
from utils import csv_loader

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

ls_data = csv_loader.load_lightship_data(['task_title_keyword_hashes.csv',
                                          'task_details_keyword_hashes.csv',
                                          'task.csv'])

hash_table = unhash_data.load_hash_tables([ls_data['task_title_keyword_hashes'],
                                           ls_data['task_details_keyword_hashes']])

def test_unhash_transform(df, ht):
    result = T.unhash_transform(df, ht)
    print(result)
