import sys
import os.path as path
import src.utils.config as confg
import src.utils.common as common
import src.utils.loading as loading
import src.transforms as transforms
import src.pipelines as pipelines
import src.utils.reporting as reporting
import gensim.downloader as gensim_api
import gensim.models as gensim_models
import numpy as np
import pickle
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split


REQUIRED_FILES = ['task_title_keyword_hashes.csv',
                  'task_details_keyword_hashes.csv',
                  'account.csv',
                  'task.csv']


def exit_fatal(error):
    print('Error:', error)
    sys.exit()


def warm_start_enabled():
    return '-warm_start' in sys.argv


def warm_start_path():
    for index, arg in enumerate(sys.argv):
        if arg == '-warm_start':
            if index + 1 < len(sys.argv):
                return sys.argv(index + 1)
            exit_fatal('Must provide a data path when using the -warm_start '
                       'flag. \nUsage: -warm_start path/to/data/')


def check_required_files(data_path, keyword_path):
    has_missing = False
    for file in REQUIRED_FILES:
        in_data_path = path.exists(data_path + file)
        in_keyword_path = path.exists(keyword_path + file)
        if not in_data_path and not in_keyword_path:
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
    keyword_path = config['keyword_path']
    # Ensure the provided data path exists
    if not path.exists(data_path):
        exit_fatal(f'Data path "{data_path}" was not found.')
    if not path.exists(keyword_path):
        exit_fatal(f'Keyword path "{keyword_path}" was not found.')
    # Ensure all required files are present
    check_required_files(data_path, keyword_path)
    # All files are present, load them
    return loading.load_lightship_data(config, REQUIRED_FILES)


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
    result = common.filter_null(input_data, 'assignee_id')
    result = common.filter_neq(result, 'creator_id', config['automated_account_id'])
    # Remove instances that belong to a target class with less than 5 classes
    mask = common.mask_low_frequency(result['assignee_id'], 5)
    result = result[mask]
    # Unhash the title and details columns
    if config['unhashing_enabled']:
        result['title'] = transforms.unhash_column(result['title'], keyword_table)
        result['details'] = transforms.unhash_column(result['details'], keyword_table)
    return result


# The pipeline that will be run when grid search is disabled
def configure_default_pipeline(kv_model, config):
    classifiers = []
    # Use standard task pipeline
    if config['task_pipe_enabled']:
        classifiers.append(('task', pipelines.pipeline_task()))
    # Use word vectorization in the final model
    if config['wordvec_pipe_enabled']:
        classifiers.append(('word-v', pipelines.pipeline_word_vectorizer(kv_model)))
    # Use count vectorization in the final model
    if config['countvec_pipe_enabled']:
        classifiers.append(('count-v', pipelines.pipeline_count_vectorizer()))
    return StackingClassifier(classifiers)


def load_cached_model(config):
    model_path = config['cache_location'] + 'model'
    if not path.exists(model_path):
        exit_fatal('No cached model avaiable for warm start mode.')
    try:
        model = pickle.load(model_path)
        return model
    except pickle.UnpicklingError:
        exit_fatal('Failed to unserialize cached model.')


def cache_model(model, config):
    model_path = config['cache_location'] + 'model'
    try:
        pickle.dump(model, model_path)
    except pickle.PickleError:
        exit_fatal('Model caching failed (unserializable object encountered).')


def main():

    # DATA LOADING PHASE
    config = confg.load()
    frames = load_required_files(config)
    input_data = frames['task']
    keyword_table = common.load_hash_tables(
        [frames['task_title_keyword_hashes'],
         frames['task_details_keyword_hashes']])
    kv_model = load_gensim_model(config)
    account_lookup = common.get_account_name_dict(frames['account'])
    
    # PREPROCESSING PHASE
    input_data = initial_preprocessing(input_data, keyword_table, config)
    
    # MODEL EVALUATION PHASE
    if warm_start_enabled():
        warm_start_evaluation(input_data, kv_model, account_lookup)
    elif config['grid_search_enabled']:
        grid_search_evaluation(input_data, kv_model, account_lookup)
    else:
        default_evaluation(input_data, kv_model, config, account_lookup)
    

def default_evaluation(input_data, kv_model, config, account_lookup):
    
    y = input_data['assignee_id']
    
    # TEST MODEL
    test_size = config['test_size']
    X_train, X_test, y_train, y_test = train_test_split(input_data, y, test_size=test_size)
    
    clf = configure_default_pipeline(kv_model, config)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    y_class = clf.classes_
    task_ids = input_data['id'].to_numpy()
    
    # REPORTING PHASE
    report_builder = reporting.ReportBuilder(min_spacing=10)
    report_builder.data_summary(input_data, config)
    report_builder.configuration(config)
    report_builder.classification(y_pred, y_test, y_prob)
    report_builder.recommendations(task_ids, y_test, y_prob, y_class, account_lookup, config)
    report_builder.print_report()
    report_builder.save_report(config['report_location'])


def warm_start_evaluation(input_data, config, account_lookup):
    
    # Load cached model
    clf = load_cached_model(config)
    # Model already trained, run test on entire set
    X_test = input_data
    y_test = input_data['assignee_id']
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    y_class = clf.classes_
    task_ids = input_data['id'].to_numpy()
    
    # REPORTING PHASE
    report_builder = reporting.ReportBuilder(min_spacing=10)
    report_builder.data_summary(input_data, config)
    report_builder.configuration(config)
    report_builder.classification(y_pred, y_test, y_prob)
    report_builder.recommendations(task_ids, y_test, y_prob, y_class, account_lookup, config)
    report_builder.print_report()
    report_builder.save_report(config['report_location'])

def grid_search_evaluation(input_data, kv_model, config, account_lookup):
    pass # todo
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    