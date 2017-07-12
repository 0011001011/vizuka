from config.references import (
    INPUT_FILE_BASE_NAME,
    DATA_PATH,
    VERSION,
    REDUCTION_SIZE_FACTOR,
    TSNE_DATA_PATH,
    PARAMS,
    OUTPUT_NAME,
)

"""


"""
import itertools

import numpy as np
import logging
from sklearn.manifold import TSNE as tsne

"""
from shared_helpers import config

DATA_VIZ_CONFIG = config.load_config(__package__)

REDUCTION_SIZE_FACTOR = DATA_VIZ_CONFIG['REDUCTION_SIZE_FACTOR']
VERSION = DATA_VIZ_CONFIG['VERSION']
PARAMS = DATA_VIZ_CONFIG['PARAMS']

DATA_PATH = DATA_VIZ_CONFIG['DATA_PATH']

INPUT_FILE_BASE_NAME = os.path.join(DATA_PATH, DATA_VIZ_CONFIG['INPUT_FILE_BASE_NAME'])
DATA_PATH            = os.path.join(DATA_PATH, DATA_VIZ_CONFIG['DATA_PATH'])
TSNE_DATA_PATH       = os.path.join(BASE_PATH, DATA_VIZ_CONFIG['TSNE_DATA_PATH'])
"""


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
        - has name "$(file_base_name)_x_y_$(version).npz"
                $(version) is for e.g "20170614"
                $(file_base_name) if for e.g 'processed_predictions' or 'one_hot'
        - contains entry x with input data
        - contains entry y_$(output_name) with output data (labels)
        - optionnaly an encoder to translate machine-readable labels to human-readable labels
                (actually it is the opposite e.g: {604600:[False, True, False, .., False]})

    :return: (input for t-SNE, classes, encoder (humantomachine class labels),
    decoder (machinetohuman labels))

    Note that encoder/decoder are assumed trivial if no encoder are found in npz

    """

    x_small = []
    y_small = []

    xy = np.load(path + INPUT_FILE_BASE_NAME + '_x_y' + VERSION + '.npz')
    x = xy['x']
    y = xy['y_' + output_name]

    x_small = x[:int(x.shape[0] / REDUCTION_SIZE_FACTOR)]; del x # noqa
    y_small = y[:int(y.shape[0] / REDUCTION_SIZE_FACTOR)]; del y # noqa
    
    logging.info(xy.keys())
    logging.info(x_small[0])
    logging.info(y_small[0])

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

    del xy

    return (np.array(x_small), np.array(y_small), class_encoder, class_decoder)


def learn_tSNE(x, params=PARAMS, version=VERSION, path=TSNE_DATA_PATH,
               reduction_size_factor=REDUCTION_SIZE_FACTOR):
    """
    Learn tSNE representation.

    :param x: input data to project
    :param params: t-SNE parameters to learn
                   learn every combination possible
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

    for perplexity, learning_rate, init, n_iter in concatenated_iterator:
        logging.info("learning model", params)
 
        param = (
            perplexity,
            learning_rate,
            init,
            n_iter
        )
        models[param] = tsne(
            perplexity=perplexity,
            learning_rate=learning_rate,
            init=init,
            n_iter=n_iter
        )
        x_transformed[param] = models[param].fit_transform(x)
 
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
            model=models[param]
        )

        logging.info("done!")
 
    return x_transformed, models


def load_tSNE(params=PARAMS, version=VERSION, path=TSNE_DATA_PATH,
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

        logging.info("RNmodel=loadin ", params)
        try:
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

            x_transformed[param] = np.load(full_path)['x_2D']

            try:
                models[param] = np.load(full_path)['model']
            except KeyError:
                logging.info("old version, model not found, only embedded data")
            logging.info("RNmodel=ready")

        except FileNotFoundError as e:
            logging.info(" not found", e)

    return x_transformed, models
