import src.utils.util_misc as util_misc

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import datetime
import os.path as path
from os import mkdir

class ReportBuilder:
    
    def __coalesce(self):
        return '\n'.join([
            self.__data_summary,
            self.__configuration,
            self.__classification,
            self.__recommendations
        ])
    
    
    def __build_section_header(self, text, length):
        leftover = length - len(text) - 1
        left, right = 0, 0
        if leftover % 2 == 0:
            left = leftover // 2
            right = left
        else:
            left = (leftover+1) // 2
            right = leftover - left
        return f'{"":*>{left-1}} {text} {"":*>{right-1}}\n'
    
    
    def __build_section(self, header_text, entries):
        lines = []
        max_key_len = max([len(key) for key in entries])
        max_key_len += self.__min_spacing
        for key in entries:
            key_len = len(key)
            pad_len = max_key_len - key_len
            if isinstance(entries[key], float):
                lines.append(f'{key}:{"":{pad_len}}{entries[key]:0.3}\n')
            else:
                lines.append(f'{key}:{"":{pad_len}}{entries[key]}\n')
        max_line_len = max([len(s) for s in lines])
        result = self.__build_section_header(header_text, max_line_len)
        result += ''.join(lines)
        return result
        
    
    def __build_recommendation(self, task_id, probs, y_true, y_class, account_lookup, config):
        recs = util_misc.get_acc_recommendations(probs, y_class, account_lookup)
        if len(recs) >= config['max_recommendations']:
            recs = recs[:config['max_recommendations']]
        result = f'Task id: {task_id}\n'
        result += f'Actual assignee: {account_lookup[y_true]}\n'
        result += 'Predictions:\n'
        max_name_len = max([len(rec['account_name']) for rec in recs])
        max_name_len += self.__min_spacing
        for rec in recs:
            name_len = len(rec['account_name'])
            pad_len = max_name_len - name_len
            result += f'  {rec["account_name"]}{"":>{pad_len}}{rec["probability"]:0.3}\n'
        return result
    
    
    def __init__(self, min_spacing=5):
        self.__min_spacing = min_spacing
        self.__data_summary = ''
        self.__configuration = ''
        self.__classification = ''
        self.__recommendations = ''
    
    
    def data_summary(self, data, config):
        size = data.shape[0]
        test_size = int(size*config['test_size'])
        train_size = size - test_size
        entries = {
            'Example count': size,
            'Test size': test_size,
            'Train size': train_size
        }
        self.__data_summary = self.__build_section('DATA SUMMARY', entries)
    
    
    def configuration(self, config):
        kv_model = config['kv_model'] if config['wordvec_pipe_enabled'] else 'N/A'
        entries = {
            'Keyword unhashing enabled': config['unhashing_enabled'],
            'Task pipeline enabled': config['task_pipe_enabled'],
            'Word vectorization enabled': config['wordvec_pipe_enabled'],
            'Count vectorization enabled': config['countvec_pipe_enabled'],
            'Word vectorization model name': kv_model
        }
        self.__configuration = self.__build_section('RUN CONFIGURATION', entries)
    
    
    def classification(self, y_pred, y_true, y_proba):
        entries = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Balanced accuracy': balanced_accuracy_score(y_true, y_pred),
            'Top 2 accuracy': top_k_accuracy_score(y_true, y_proba, k=2),
            'Top 3 accuracy': top_k_accuracy_score(y_true, y_proba, k=3),
            'F1 score (macro)': f1_score(y_true, y_pred, average='macro'),
            'Roc auc (ovo)': roc_auc_score(y_true, y_proba, multi_class='ovo')
        }
        self.__classification = self.__build_section('CLASSIFICATION RESULTS', entries)
    
    
    def recommendations(self, task_ids, y_true, y_proba, y_class, account_lookup, config):
        results = []
        for i, actual in enumerate(y_true):
            result = self.__build_recommendation(task_ids[i],
                                                 y_proba[i],
                                                 actual,
                                                 y_class,
                                                 account_lookup,
                                                 config)
            results.append(result)
        header = self.__build_section_header('RECOMMENDATIONS', 50)
        self.__recommendations = header + '\n'.join(results)
    
    
    def print_report(self):
        print(self.__coalesce())
    
    
    def save_report(self, out_dir):
        if not path.exists(out_dir):
            mkdir(out_dir)
        timestamp = datetime.datetime.now()
        timestamp = timestamp.strftime('%Y%m%dT%H%M%S')
        filepath = out_dir + 'Report ' + timestamp + '.txt'
        with open(filepath, mode='w') as f:
            f.write(self.__coalesce())
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        