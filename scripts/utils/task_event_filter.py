## task_event_filter.py
##
## Author: Cody Painter
## Date: 2022/10/09
##
## Description:
## Filter events by dtype

import pandas as pd
import numpy as np
from csv_loader import load_lightship_data


def filter_task_events(task_event_df, event_dtype, cols):
    fltr = (task_event_df['dtype'] == event_dtype)
    return pd.DataFrame(task_event_df[fltr], columns=cols)



################
## Example usage
################

##ls_data = load_lightship_data('task_event.csv')
##filtered = filter_task_events(ls_data['task_event'],
##                              'CommentEvent',
##                              ['task_id', 'content'])
##print(filtered.head)
