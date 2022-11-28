import sys
import os.path as path
import src.utils.config
import src.utils.csv_loader as csv_loader
import src.utils.unhash_data as unhash
import src.utils.util_misc as util_misc
import src.transforms as transforms
import src.pipelines_refactored as pipelines
import gensim.downloader as gensim_api
import gensim.models as gensim_models
import numpy as np
import pickle
import sklearn.metrics as metrics
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score, train_test_split



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


# Generate a binary mask to filter out target classes with a total count 
# less than n.
def mask_low_frequency(data, n):
    target = data['assignee_id']
    vals, counts = np.unique(target, return_counts=True)
    return np.isin(target, vals[counts >= n])


def initial_preprocessing(input_data, keyword_table, config):
    # Filter null assignees and auto-generated tasks
    result = input_data[input_data['assignee_id'].notnull()]
    result = result[result['creator_id'].ne(config['automated_account_id'])]
    # Remove instances that belong to a target class with less than 5 classes
    result = result[mask_low_frequency(result, 5)]
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


def report_configuration(config):
    print('*** RUN CONFIGURATION ***')
    print(f'Keyword unhashing enabled            {config["unhashing_enabled"]}')
    print(f'Task pipeline enabled                {config["task_pipe_enabled"]}')
    print(f'Word vectorization pipeline enabled  {config["wordvec_pipe_enabled"]}')
    print(f'Count vectorization pipeline enabled {config["countvec_pipe_enabled"]}')
    if config['wordvec_pipe_enabled']:
        print(f'Word vectorization model name: {config["kv_model"]}')
    print()

def report_input_data(input_data, test_size):
    print('*** INPUT DATA ***')
    total_examples = input_data.shape[0]
    test_examples = int(total_examples * test_size)
    train_examples = total_examples - test_examples 
    print(f'Example count:      {total_examples}')
    print(f'Test size:          {test_examples}')
    print(f'Train size:         {train_examples}')
    print()


def report_classification(y_pred, y_true, y_prob):
    print('*** CLASSIFICATION RESULT ***')
    acc = metrics.accuracy_score(y_true, y_pred)
    bal_acc = metrics.balanced_accuracy_score(y_true, y_pred)
    top_2 = metrics.top_k_accuracy_score(y_true, y_prob, k=2)
    top_3 = metrics.top_k_accuracy_score(y_true, y_prob, k=3)
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    roc_auc = metrics.roc_auc_score(y_true, y_prob, multi_class='ovo')
    print(f'Accuracy:           {acc}')
    print(f'Balanced accuracy   {bal_acc}')
    print(f'Top 2 accuracy:     {top_2}')
    print(f'Top 3 accuracy:     {top_3}')
    print(f'F1 score (macro):   {f1_macro}')
    print(f'Roc auc (ovo):      {roc_auc}')
    print()


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
    
    # MODEL EVALUATION PHASE
    if warm_start_enabled():
        warm_start_evaluation(input_data, kv_model, config)
    elif config['grid_search_enabled']:
        grid_search_evaluation(input_data, kv_model, config)
    else:
        default_evaluation(input_data, kv_model, config)
    

def default_evaluation(input_data, kv_model, config):
    
    y = input_data['assignee_id']
    
    # TEST MODEL
    test_size = config['test_size']
    X_train, X_test, y_train, y_test = train_test_split(input_data, y, test_size=test_size)
    
    clf = configure_default_pipeline(kv_model, config)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    
    # REPORTING PHASE
    report_input_data(input_data, test_size)
    report_configuration(config)
    report_classification(y_pred, y_test, y_prob)


def warm_start_evaluation(input_data, config):
    
    # Load cached model
    clf = load_cached_model(config)
    # Model already trained, run test on entire set
    X_test = input_data
    y_test = input_data['assignee_id']
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    
    # REPORTING PHASE
    report_input_data(input_data, 1)
    report_configuration(config)
    report_classification(y_pred, y_test, y_prob)
    

def grid_search_evaluation(input_data, kv_model, config):
    pass # todo
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    