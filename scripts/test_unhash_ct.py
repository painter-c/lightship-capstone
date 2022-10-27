import pipelines as pl
import utils.csv_loader as loader
import utils.unhash_data as uhd

ls_data = loader.load_lightship_data(['task_title_keyword_hashes.csv',
                                          'task_details_keyword_hashes.csv',
                                          'task.csv'])

hash_table = uhd.load_hash_tables([ls_data['task_title_keyword_hashes'],
                                   ls_data['task_details_keyword_hashes']])

kw_df = ls_data['task'][['title', 'details']]
ct = pl._build_ct_unhash(hash_table)
result_df = ct.fit_transform(kw_df)

print(result_df[0])
