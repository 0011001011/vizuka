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
#from MulticoreTSNE import MulticoreTSNE as tsne
from sklearn.manifold import TSNE as tsne

from vizuka.config import (
    INPUT_FILE_BASE_NAME,
    DATA_PATH,
    VERSION,
    REDUCTION_SIZE_FACTOR,
    REDUCTED_DATA_PATH,
    PARAMS_LEARNING,
    OUTPUT_NAME,
    PCA_MIN_VARIANCE,
)



def reduce_with_PCA(x, variance_needed=PCA_MIN_VARIANCE):
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


def learn_tSNE(x, params=PARAMS_LEARNING, version=VERSION, path=REDUCTED_DATA_PATH,
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
        x = reduce_with_PCA(x, variance_needed=pca_variance_needed)

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
            # only use with Multicore_tSNE:   n_jobs=12,
        ).fit_transform(x)
        logging.info("done!")
 
        name = ''.join('_' + str(p) for p in param)
        full_path = ''.join([
            path,
            'embedded_x_1-',
            str(reduction_size_factor),
            name,
            version,
            '.npz',
        ])
 
        np.savez(
            full_path,
            x_2D=x_transformed[param],
        )  # model=models[param])
 
    return x_transformed, models


def load_tSNE(params=PARAMS_LEARNING, version=VERSION, path=REDUCTED_DATA_PATH,
              reduction_size_factor=REDUCTION_SIZE_FACTOR):
    """
    Load tSNE representation.

    :param params: dict of multiples t-SNE parameters of the representations
    we want
                   load every combination available
                   .. seealso:: sklearn.manifold.TSNE()
    :type params: {
                    'perplexities':array(int),
                    'learning_rates':array(int),
                    'inits':array({'pca', 'random'})
                    'n_iter':array(int),
                    }
    :param version: version of data to load (e.g: _20170614)
    :param path: location of the 2D representation to load
    :param reduction_size_factor: factor by which the number of samples
    is divided (tsne is greedy)

    :return: Embedded data in 2D space, and t-SNE model
    :rtype:  dict{params:(float,float)}, dict({params:tsne.model})
    """

    perplexities = params['perplexities']
    learning_rates = params['learning_rates']
    inits = params['inits']
    n_iters = params['n_iters']

    x_transformed = {}
    models = {}

    for perplexity, learning_rate, init, n_iter in itertools.product(
            perplexities, learning_rates, inits, n_iters):

        param = (perplexity, learning_rate, init, n_iter)
        name = ''.join('_' + str(p) for p in param)
        full_path = ''.join([
            path,
            'embedded_x_1-',
            str(reduction_size_factor),
            name,
            version,
            '.npz',
        ])
        logging.info("RNmodel=trying to load %s %s %s %s from %s",
                     str(perplexity),
                     str(learning_rate),
                     str(init),
                     str(n_iter),
                     full_path,
                     )
        if os.path.exists(full_path):
            x_transformed[param] = np.load(full_path)['x_2D']
        
            try:
                models[param] = np.load(full_path)['model']
            except KeyError:
                logging.info("old version, model not found, only embedded data")
            logging.info("RNmodel=ready")
        else:
            logging.info("model {} {} {} {}  not found".format(
                perplexity,
                learning_rate,
                init,
                n_iter
                )
            )


    return x_transformed, models


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    x, y, encoder, decoder = load_raw_data()
    x_transformed, models = learn_tSNE(x)

