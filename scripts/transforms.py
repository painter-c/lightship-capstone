import numpy as np

def dayofweek_transform(df):
    out = []
    for name in df.columns:
        col = df[name]
        if not str(col.dtype).count('datetime64'):
            continue
        new_col = col.dt.dayofweek
        new_col = new_col.to_numpy()
        new_col = new_col.reshape(-1, 1)
        if len(out):
            out = np.hstack((out, new_col))
        else:
            out = new_col
    return out
            
def timeofday_transform(df):
    out = []
    for name in df.columns:
        col = df[name]
        if not str(col.dtype).count('datetime64'):
            continue
        new_col = col.dt.hour
        new_col = new_col.to_numpy()
        new_col = new_col.reshape(-1, 1)
        if len(out):
            out = np.hstack((out, new_col))
        else:
            out = new_col
    return out