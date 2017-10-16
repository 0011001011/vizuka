"""
Here is the code summoned to reduce the dimension of your
precious data, and also to load it.
We use t-SNE and, if you want, PCA just before it.

..note:: tSNE from sklearn is not the best but is standard
I suggest you to uncomment 'from MulticoreTSNE import TSNE as tsne'
as it will be much faster and won't crash if you use too much RAM.
However this needs extra-install steps :
-> cf https://github.com/DmitryUlyanov/Multicore-TSNE
"""

import itertools
import os
import logging

import numpy as np
from sklearn.decomposition import PCA

from vizuka.config import (
    INPUT_FILE_BASE_NAME,
    DATA_PATH,
    VERSION,
    REDUCTION_SIZE_FACTOR,
    REDUCED_DATA_PATH,
    REDUCED_DATA_NAME,
    PARAMS_LEARNING,
    PCA_MIN_VARIANCE,
)


def PCA_reduce(x, variance_needed=PCA_MIN_VARIANCE):
    """
    Reduce your dataset x with PCA
    variance_needed: how much of the original variance you want to capture
    """
    
    logging.info("starting PCA dimensional reduction, needs an explained variance of {}%".format(
        variance_needed*100)
        )
    pca = PCA(svd_solver='randomized')
    pca.fit(x)

    nb_dim_to_keep = 0
    variance_explained = 0

    while variance_explained < variance_needed:
        variance_explained += pca.explained_variance_ratio_[nb_dim_to_keep]
        nb_dim_to_keep += 1

    x_pcaed = pca.fit_transform(x)
    x_reduced = x_pcaed[:,:nb_dim_to_keep]
    
    logging.info("PCA successfull, {} dimensions (axis) where kept after orthnormalization".format(
        nb_dim_to_keep)
        )

    return x_reduced
