import pandas as pd

DATA_PATH = '../data/set_1/'

def load_lightship_data(files='all'):
    
    lightship_data = {}

    if files == 'all' or 'account.csv' in files:
        lightship_data['account'] = pd.read_csv(
            DATA_PATH + 'account.csv',
            parse_dates=[1,2])

    if files == 'all' or 'membership.csv' in files:
        lightship_data['membership'] = pd.read_csv(
            DATA_PATH + 'membership.csv',
            parse_dates=[1,2])

    if files == 'all' or 'mention.csv' in files:
        lightship_data['mention'] = pd.read_csv(
            DATA_PATH + 'mention.csv')

    if files == 'all' or 'task.csv' in files:
        lightship_data['task'] = pd.read_csv(
            DATA_PATH + 'task.csv',
            parse_dates=[1,2,8])

    ## Todo: load other tables

    if files == 'all' or 'task_teams.csv' in files:
        lightship_data['task_teams'] = pd.read_csv(
            DATA_PATH + 'task_teams.csv')

    ## Todo: load other tables

    return lightship_data
                    
