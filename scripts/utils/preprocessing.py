# preprocessing.py
#
# Author: Cody Painter
# Date: October 1, 2022
#
# Description:
# Each function represents a data preprocessing pipeline. Functions write a pickle object
# to output_path and return the preprocessed data frame.

import pandas as pd
import numpy as np

def preprocess_expand_teams(path_to_data):
    return None

def preprocess_expand_observers(path_to_data):
    return None

# Base task data preprocessing pipeline
# Performs preprocessing steps that are common to all models
# Includes:
#   - load csv data
#   - remove useless columns
def preprocess_task_base(path_to_data):
    # define data types for parsing
    task_dtypes = {'priority':int, 'project_id':str, 'assignee_id':str, \
                   'creator_id':str, 'title':str, 'details':str}
    # read csv data
    df = pd.read_csv(path_to_data + 'set_1/task.csv', dtype=task_dtypes, parse_dates=[1])
    # remove useless columns
    df.drop(['id','modified','confidential','due_date','state','state_modified','numerator'], axis=1, inplace=True)
    
    return df

# Multinomial logistic regression
# Version 1
#
def preprocess_mlregression_v1(path_to_data, output_path=None):
    # load the data
    df = preprocess_task_base(path_to_data)
    
    # drop additional columns
    df.drop(['title','details'], axis=1, inplace=True)
    
    # Transform the created date column to days of the week
    df['day_of_week'] = df['created'].dt.day_name()
    df.drop('created', axis=1, inplace=True)
    
    # Transform day_of_week to dummy columns
    dummies1 = pd.get_dummies(df['day_of_week'],prefix='dayofweek',drop_first=True)
    df.drop('day_of_week', axis=1, inplace=True)
    
    # turn project_id to dummy columns
    dummies2 = pd.get_dummies(df['project_id'],prefix='project_id',drop_first=True)
    df.drop('project_id', axis=1, inplace=True)
    
    # turn assignee_id to dummy columns
    dummies3 = pd.get_dummies(df['assignee_id'],prefix='assignee_id',drop_first=True)
    df.drop('assignee_id', axis=1, inplace=True)
    
    # turn day_of_week to dummy columns
    dummies4 = pd.get_dummies(df['creator_id'],prefix='creator_id',drop_first=True)
    df.drop('creator_id', axis=1, inplace=True)

    df = pd.concat([df, dummies1, dummies2, dummies3, dummies4])
    
    return df
    
