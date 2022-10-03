import utils.preprocessing as prep

DATA_PATH = '../data/'

df = prep.preprocess_mlregression_v1(DATA_PATH)

df.info()
