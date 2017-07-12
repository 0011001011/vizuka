import os

BASE_PATH       = '/home/sofian/ovh/'

DATA_PATH       = os.path.join(BASE_PATH, 'data/set/')  
TSNE_DATA_PATH  = os.path.join(BASE_PATH, 'data/set/tSNE/')
MODEL_PATH      = os.path.join(BASE_PATH, 'data/models/')
GRAPH_PATH      = os.path.join(BASE_PATH, 'graph/')

# File containing data to be t-SNEed
INPUT_FILE_BASE_NAME = 'one_hot'

# A version is a string added to the end of each filename
VERSION = '_20170614'

# default RN for predictions
DEFAULT_RN = 'one_hot_1000-600-300-200-100_RN'

# data output name (labels)
OUTPUT_NAME = 'account'

# t-SNE parameters
# best tuple so far is (50,1000,pca,15000)
PARAMS = { 'perplexities'  : [30,40,50],
           'learning_rates': [500, 800, 1000],
           'inits'         : ['random', 'pca'],
           'n_iters'       : [5000, 15000]
         }

# 30 for OVH, 50 for local, 15 for epinal
REDUCTION_SIZE_FACTOR = 15

# False -> load representation
# True  -> find new projection for t-SNE
DO_CALCULUS = False
