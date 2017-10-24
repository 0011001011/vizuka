import numpy as np
import logging

try:
    from MulticoreTSNE import MulticoreTSNE as tsne_algorithm
    using_multicore_tsne = True
    multicore_parameters = {'n_jobs':3}
    logging.info('dimension_reduction = Using fast tSNE')

except:
    from sklearn.manifold import TSNE as tsne_algorithm
    using_multicore_tsne = False
    multicore_parameters = {}
    logging.info('dimension_reduction = Using slow tSNE (see requirements/requirements.apt')

from vizuka.dimension_reduction import (
        projector,
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
            **multicore_parameters,
            )
        self.projections = []

    def project(self, x):
        self.projections = self.engine.fit_transform(x)
        return self.projections
