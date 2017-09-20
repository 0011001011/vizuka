"""
Here are the default parameters used in all the package
There are 1.path 2.filenames 3.learning parameters
"""
import os

#
# ALL DEFAULT PATH
#

BASE_PATH       = os.path.join('data/')

DATA_PATH       = os.path.join(BASE_PATH, 'set/')
REDUCTED_DATA_PATH  = os.path.join(BASE_PATH, 'reducted/')
MODEL_PATH      = os.path.join(BASE_PATH, 'models/')
GRAPH_PATH      = os.path.join(BASE_PATH, 'graph/')


#
# ALL DEFAULT FILENAME
#

# File containing data to be t-SNEed
INPUT_FILE_BASE_NAME = 'preprocessed_inputs'
RAW_NAME = 'originals'

# default RN for predictions
DEFAULT_PREDICTOR = 'metapredict'

# A version is a string added to the end of each filename
VERSION = '_20170921'

# data output name (labels)
OUTPUT_NAME = 'account'


#
# ALL LEARNING PARAMETERS
#

# t-SNE parameters
# best tuple so far is (50,1000,pca,15000)
PARAMS_LEARNING = { 'perplexities'  : [30,40,50,80],
                                         # roughly the number of neighbors in cluster
                                         # https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf
                                         # p4
           'learning_rates': [500, 800, 1000],
           'inits'         : ['random'], #deprecated, use pca_variance_needed instead
           'n_iters'       : [5000, 15000]
         }

# t-SNE parameters for the reduced data we will draw
PARAMS_VIZ = { 'perplexity'  : 80,
           'learning_rate': 1000,
           'init'         : 'random', #deprecated, use pca_variance_needed instead
           'n_iter'       : 15000,
           }

PCA_DIMS = 42  # ~90% of explained_variance in test case

# 30 for OVH, 50 for local, 15 for epinal
REDUCTION_SIZE_FACTOR = 1
