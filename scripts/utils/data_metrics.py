# data_metrics.py
#
# Author: Cody Painter
# Date:   October 1, 2022
#
# Description
# Utitily functions for data analysis using Pandas DataFrames

import sys
import pandas as pd
import numpy as np

def count_uniques(df):
    print('UNIQUE VALUES:')
    cname = df.columns
    for cname in df.columns:
        series = df.loc[:,cname]
        print(series.name, '->', series.nunique())



# from sklearn.datasets import load_iris

# def test():
    # df, _ = load_iris(return_X_y=True, as_frame=True)
    # count_uniques(df)
    
    
# test()