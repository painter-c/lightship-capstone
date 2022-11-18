from collections import namedtuple

# Returns a dictionary that maps account ids to names
def get_account_name_dict(acc_df):
    acc_dict = {}
    for row in acc_df.itertuples():
        acc_dict[row.id] = row.name
    return acc_dict


_recc = namedtuple('_recc', 'probability account_id name')

# Takes a list of probabilities and account ids and returns
# a sorted list of of tuples of the form (prob, id, name).
def get_acc_reccomendations(ps, ids, acc_dict):
    reccs = [_recc(ps[i], ids[i], acc_dict[ids[i]]) for i in range(len(ps))]
    reccs.sort(key=lambda k: k.probability, reverse=True)
    return reccs