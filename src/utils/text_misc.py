from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def tokenize_all(X):
    all_tokens = []
    n_cols = X.shape[1]
    for i in range(n_cols):
        col = X[:,i]
        for item in col:
            if not item:
                continue
            tokens = word_tokenize(item)
            all_tokens.extend(tokens)
    return all_tokens

def count_stopwords(X):
    tokens = tokenize_all(X)
    stop = set(stopwords.words('english'))
    stop_count = 0
    for token in tokens:
        if token in stop:
            print(token)
            stop_count += 1
    return stop_count

def extract_words_by_tag(text, tag, spacy_nlp):
    if text is None:
        return ''
    doc = spacy_nlp(str(text))
    words = []
    for word in doc:
        if word.pos_.lower() == tag.lower():
            words.append(word.text)
    return ' '.join(words)

def remove_duplicates(text):
    if text is None:
        return ''
    return ' '.join(list(set(text.split())))

def join_string(a, b):
    if a is None and b is not None:
        return b
    elif b is None and a is not None:
        return a
    elif a is None and b is None:
        return ''
    else:
        return ' '.join((a, b))