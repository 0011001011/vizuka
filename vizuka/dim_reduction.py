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
    PCA_DIMS,
)


def load_raw_data(
        file_base_name=INPUT_FILE_BASE_NAME,
        output_name=OUTPUT_NAME,
        path=DATA_PATH,
        version=VERSION,
        reduction_factor=REDUCTION_SIZE_FACTOR
):
    """
    Loads and returns the data for tSNE
    One-hotted and regular ML-encoding

    File to load should :
        - be in path
        - has name "(INPUT_FILE_BASE_NAME)_x_y_(VERSION).npz"
                (VERSION) is for e.g "_20170614"
                (FILE_BASE_NAME) if for e.g 'processed_predictions' or 'one_hot'
        - contains entry x with input data
        - contains entry y_(OUTPUT_NAME) with output data (possible_outputs_list to predict)
        - optionnaly an encoder to translate machine-readable possible_outputs_list to human-readable possible_outputs_list
                (actually it is the opposite e.g: {604600:[False, True, False, .., False]})

    :return: (input for t-SNE, classes, encoder (humantomachine class possible_outputs_list),
    decoder (machinetohuman possible_outputs_list))

    Note that encoder/decoder are assumed trivial if no encoder are found in npz

    """
    
    x_small = []
    y_small = []
    
    xy = np.load(path + INPUT_FILE_BASE_NAME + VERSION + '.npz')

    if output_name + '_encoder' in xy.keys():
        logging.info("found encoder")
        # I don't understant either but it is a 0-d array (...)
        class_encoder = xy[output_name + '_encoder'][()]
        decoder_dic = {class_encoder[k].argsort(
        )[-1]: k for k in class_encoder.keys()}

        class_decoder = lambda oh: decoder_dic[np.argsort(oh)[-1]] # noqa
    else:
        class_encoder = {y: y for y in set(y_small)}
        class_decoder = lambda x: x # noqa

    x = xy['x']
    x_small = x[:int(x.shape[0] / reduction_factor)]; del x # noqa

    if 'y_' + output_name in xy.keys():
        y = xy['y_' + output_name]
        del xy
        return (np.array(x_small), np.array(y), class_encoder, class_decoder)
    elif 'y_' + output_name + '_decoded':
        y_decoded = xy['y_' + output_name + '_decoded']
        del xy
        return (np.array(x_small), np.array(y_decoded), class_encoder, class_decoder)
        # y = np.array([class_encoder[obs] for obs in y_decoded])

    #y_small = y[:int(y.shape[0] / reduction_factor)]; del y # noqa

def reduce_with_PCA(x, variance_needed):
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
    
    print(params)
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

        logging.info("RNmodel=trying to load %s %s %s %s",
                     str(perplexity),
                     str(learning_rate),
                     str(init),
                     str(n_iter),
                     )
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

