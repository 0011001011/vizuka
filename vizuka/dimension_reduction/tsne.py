import itertools
import os
import logging

import numpy as np
# from MulticoreTSNE import MulticoreTSNE as tsne_algorithm
from sklearn.manifold import TSNE as tsne_algorithm

from vizuka.config import (
    VERSION,
    REDUCED_DATA_PATH,
    REDUCED_DATA_NAME,
)
from vizuka.dimension_reduction import (
        projector,
        tsne,
        )


class tSNE(projector.Projector):

    def __init__(self, perplexity, learning_rate, n_iter):
        
        self.perplexity     = perplexity
        self.learning_rate  = learning_rate
        self.n_iter         = n_iter

        self.parameters = [perplexity, learning_rate, n_iter]

        self.engine = tsne_algorithm(
            perplexity    = self.perplexity,
            learning_rate = self.learning_rate,
            n_iter        = self.n_iter,
            # n_jobs=3, # only use with Multicore_tSNE !
            )
        self.projections = []

    def project(self, x):
        self.projections = self.engine.fit_transform(x)
        return y
