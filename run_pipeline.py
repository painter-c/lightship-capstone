import sys
import os.path as path
import src.utils.config
import src.utils.csv_loader as csv_loader
import src.utils.unhash_data as unhash
import src.utils.util_misc as util_misc
import src.transforms as transforms
import gensim.downloader as gensim_api
import gensim.models as gensim_models


REQUIRED_FILES = ['task_title_keyword_hashes.csv',
                  'task_details_keyword_hashes.csv',
                  'account.csv',
                  'task.csv']


def exit_fatal(error):
    print('Error:', error)
    exit()


def warm_start_enabled():
    return '-warm_start' in sys.argv


def warm_start_path():
    for index, arg in enumerate(sys.argv):
        if arg == '-warm_start':
            if index + 1 < len(sys.argv):
                return sys.argv(index + 1)
            exit_fatal('Must provide a data path when using the -warm_start '
                       'flag. \nUsage: -warm_start path/to/data/')


def check_required_files(data_path):
    has_missing = False
    for file in REQUIRED_FILES:
        if not path.exists(data_path + file):
            print(f'Missing file: {data_path+file}')
            has_missing = True
    if has_missing:
        exit_fatal(f'Error: there were missing files at "{data_path}"')


def get_data_path(config):
    if warm_start_enabled():
        return warm_start_path()
    else:
        return config['data_path']


def load_required_files(config):
    data_path = get_data_path(config)
    # Ensure the provided data path exists
    if not path.exists(data_path):
        exit_fatal(f'Data path "{data_path}" was not found.')
    # Ensure all required files are present
    check_required_files(data_path)
    # All files are present, load them
    return csv_loader.load_lightship_data(data_path, REQUIRED_FILES)


def load_gensim_model(config):
    model_path = config['cache_location'] + config['kv_model'] + '.kv'
    if path.exists(model_path):
        return gensim_models.KeyedVectors.load(model_path)
    # No cached model found, attempt to download using the gensim api
    try:
        model = gensim_api.load(config['kv_model'])
        model.save(model_path)
        return model
    except Exception:
        exit_fatal(f'"{config["kv_model"]}" is not a valid Gensim model.')


def initial_preprocessing(input_data, keyword_table, config):
    # Filter null assignees and auto-generated tasks
    result = input_data[input_data['assignee_id'].notnull()]
    result = result[result['creator_id'].ne(config['automated_account_id'])]
    # Unhash the title and details columns
    if config['unhashing_enabled']:
        result['title'] = transforms.unhash_column(result['title'], keyword_table)
        result['details'] = transforms.unhash_column(result['details'], keyword_table)
    return result


def main():

    # DATA LOADING PHASE
    config = src.utils.config.load()
    frames = load_required_files(config)
    input_data = frames['task']
    keyword_table = unhash.load_hash_tables(
        [frames['task_title_keyword_hashes'],
         frames['task_details_keyword_hashes']])
    kv_model = load_gensim_model(config)
    account_lookup = util_misc.get_account_name_dict(frames['account'])
    
    # PREPROCESSING PHASE
    input_data = initial_preprocessing(input_data, keyword_table, config)
    
    print(input_data.head())


if __name__ == '__main__':
    main()