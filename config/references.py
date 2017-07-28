import os

BASE_PATH       = '/home/sofian/'

DATA_PATH       = os.path.join(BASE_PATH, 'data/set/')  
TSNE_DATA_PATH  = os.path.join(BASE_PATH, 'data/set/tSNE/')
MODEL_PATH      = os.path.join(BASE_PATH, 'data/models/')
GRAPH_PATH      = os.path.join(BASE_PATH, 'graph/')

# File containing data to be t-SNEed
INPUT_FILE_BASE_NAME = 'one_hot'

# A version is a string added to the end of each filename
VERSION = '_20170614'

# default RN for predictions
DEFAULT_PREDICTOR = 'metapredict.npz'

# data output name (labels)
OUTPUT_NAME = 'account'

# t-SNE parameters
# best tuple so far is (50,1000,pca,15000)
"""
PARAMS = { 'perplexities'  : [30,40,50],
           'learning_rates': [500, 800, 1000],
           'inits'         : ['random', 'pca'],
           'n_iters'       : [5000, 15000]
         }
"""

PARAMS = { 'perplexities'  : [40],
           'learning_rates': [1000],
           'inits'         : ['pca'],
           'n_iters'       : [15000]
         }

# 30 for OVH, 50 for local, 15 for epinal
REDUCTION_SIZE_FACTOR = 3

# False -> load representation
# True  -> find new projection for t-SNE
DO_CALCULUS = False
