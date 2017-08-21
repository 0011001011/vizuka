import os

BASE_PATH       = os.path.join('data/')

DATA_PATH       = os.path.join(BASE_PATH, 'set/')
TSNE_DATA_PATH  = os.path.join(BASE_PATH, 'tSNE/')
MODEL_PATH      = os.path.join(BASE_PATH, 'models/')
GRAPH_PATH      = os.path.join(BASE_PATH, 'graph/')


# File containing data to be t-SNEed
INPUT_FILE_BASE_NAME = 'preprocessed_inputs'
RAW_NAME = 'originals'

# default RN for predictions
DEFAULT_PREDICTOR = 'metapredict'

# A version is a string added to the end of each filename
VERSION = '_20170817'

# data output name (labels)
OUTPUT_NAME = 'account'

# t-SNE parameters
# best tuple so far is (50,1000,pca,15000)
PARAMS_LEARNING = { 'perplexities'  : [30,40,50,80],
                                         # roughly the number of neighbors in cluster
                                         # https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf
                                         # p4
           'learning_rates': [500, 800, 1000],
           'inits'         : ['random', 'pca'],
           'n_iters'       : [5000, 15000]
         }

PARAMS_VIZ = { 'perplexity'  : 80,
           'learning_rate': 1000,
           'init'         : 'random',
           'n_iter'       : 15000,
           }

PCA_DIMS = 42  # ~90% of explained_variance

# 30 for OVH, 50 for local, 15 for epinal
REDUCTION_SIZE_FACTOR = 1

# False -> load representation
# True  -> find new projection for t-SNE
DO_CALCULUS = False


