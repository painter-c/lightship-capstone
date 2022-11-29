import pandas as pd

def load_lightship_data(config, files='all'):
    
    data_path = config['data_path']
    keyword_path = config['keyword_path']
    
    lightship_data = {}

    if files == 'all' or 'account.csv' in files:
        lightship_data['account'] = pd.read_csv(
            data_path + 'account.csv',
            parse_dates=[1,2])

    if files == 'all' or 'membership.csv' in files:
        lightship_data['membership'] = pd.read_csv(
            data_path + 'membership.csv',
            parse_dates=[1,2])

    if files == 'all' or 'mention.csv' in files:
        lightship_data['mention'] = pd.read_csv(
            data_path + 'mention.csv')

    if files == 'all' or 'task.csv' in files:
        lightship_data['task'] = pd.read_csv(
            data_path + 'task.csv',
            parse_dates=[1,2,8],
            dtype={'title':str, 'details':str})

    if files == 'all' or 'task_event.csv' in files:
        lightship_data['task_event'] = pd.read_csv(
            data_path + 'task_event.csv',
            parse_dates=[2,3])

    if files == 'all' or 'task_event_observers.csv' in files:
        lightship_data['task_event_observers'] = pd.read_csv(
            data_path + 'task_event_observers.csv')

    if files == 'all' or 'task_event_old_observers.csv' in files:
        lightship_data['task_event_old_observers'] = pd.read_csv(
            data_path + 'task_event_old_observers.csv')
            
    if files == 'all' or 'task_event_old_teams.csv' in files:
        lightship_data['task_event_old_teams'] = pd.read_csv(
            data_path + 'task_event_old_teams.csv')
            
    if files == 'all' or 'task_event_teams.csv' in files:
        lightship_data['task_event_teams'] = pd.read_csv(
            data_path + 'task_event_teams.csv')

    if files == 'all' or 'task_observers.csv' in files:
        lightship_data['task_observers'] = pd.read_csv(
            data_path + 'task_observers.csv')

    if files == 'all' or 'task_teams.csv' in files:
        lightship_data['task_teams'] = pd.read_csv(
            data_path + 'task_teams.csv')

    if files == 'all' or 'team.csv' in files:
        lightship_data['team'] = pd.read_csv(
            data_path + 'team.csv',
            parse_dates=[1,2])

    if files == 'all' or 'task_comment_event_keyword_hashes.csv' in files:
        lightship_data['task_comment_event_keyword_hashes'] = pd.read_csv(
            keyword_path + 'task_comment_event_keyword_hashes.csv')
            
    if files == 'all' or 'task_details_keyword_hashes.csv' in files:
        lightship_data['task_details_keyword_hashes'] = pd.read_csv(
            keyword_path + 'task_details_keyword_hashes.csv')

    if files == 'all' or 'task_title_keyword_hashes.csv' in files:
        lightship_data['task_title_keyword_hashes'] = pd.read_csv(
            keyword_path + 'task_title_keyword_hashes.csv')

    return lightship_data
                    
