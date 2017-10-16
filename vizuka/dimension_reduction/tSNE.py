import itertools
import os
import logging

import numpy as np
# from MulticoreTSNE import MulticoreTSNE as tsne
from sklearn.manifold import TSNE as tsne

from vizuka import dimension_reduction
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


def tSNE_reduce(x, params=PARAMS_LEARNING, version=VERSION, path=REDUCED_DATA_PATH,
               reduction_size_factor=REDUCTION_SIZE_FACTOR, pca_variance_needed=0.9):
    """
    Learn tSNE representation.

    :param x: input data to project
    :param params: t-SNE parameters to learn
                   it will learn every combination possible
                   .. seealso:: sklearn.manifold.TSNE()
    :type params: {
                    'perplexities':array(int),
                    'learning_rates':array(int),
                    'inits':array({'pca', 'random'})
                    'n_iter':array(int),
                   }

    :param version: version of data to load (e.g: _20170614)
    :param path: where to store 2D representation, absolute path
    :param reduction_size_factor: factor by which we divide the
    number of samples (tsne is greedy)

    :return: Embedded data in 2D space, and t-SNE model
    :rtype:  dict{params:(float,float)}, dict({params:tsne.model})
    """
    
    perplexities = params['perplexities']
    learning_rates = params['learning_rates']
    inits = params['inits']
    n_iters = params['n_iters']

    models, x_transformed = {}, {}

    concatenated_iterator = itertools.product(
        perplexities,
        learning_rates,
        inits,
        n_iters
    )
    
    if pca_variance_needed:
        x = dimension_reduction.PCA.PCA_reduce(x, variance_needed=pca_variance_needed)

    for perplexity, learning_rate, init, n_iter in concatenated_iterator:
 
        param = (
            perplexity,
            learning_rate,
            init,
            n_iter
        )
        '''
        models[param] = tsne(
            perplexity=perplexity,
            learning_rate=learning_rate,
            init=init,
            n_iter=n_iter
        )'''  # in a desperate move to save RAM
        logging.info("learning model %s %s %s %s", str(perplexity), str(learning_rate), str(init), str(n_iter))

        x_transformed[param] = tsne(
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            # n_jobs=3, # only use with Multicore_tSNE:
            ).fit_transform(x)
        logging.info("done!")
 
        name = ''.join('_' + str(p) for p in param)
        full_path = ''.join([
            path,
            REDUCED_DATA_NAME,
            str(reduction_size_factor),
            name,
            '_',
            version,
            '.npz',
        ])
 
        np.savez(
            full_path,
            x_2D=x_transformed[param],
        )  # model=models[param])
 
    return x_transformed, models
