import src.utils.config as config

from src.pipelines import _build_ct_unhash
from src.utils.csv_loader import load_lightship_data
from src.utils.unhash_data import load_hash_tables

cfg = config.load()

ls_data = load_lightship_data(cfg['datasets']['set_1'],
                              ['task_title_keyword_hashes.csv',
                               'task_details_keyword_hashes.csv',
                               'task.csv'])

hash_table = load_hash_tables([ls_data['task_title_keyword_hashes'],
                               ls_data['task_details_keyword_hashes']])

kw_df = ls_data['task'][['title', 'details']]

ct = _build_ct_unhash(hash_table)
result_df = ct.fit_transform(kw_df)

print(result_df[:10])