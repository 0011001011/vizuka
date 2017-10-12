import logging
import os
import itertools

import numpy as np

from vizuka.config import (
        MODEL_PATH,
        VERSION,
        DEFAULT_PREDICTOR,
        PARAMS_LEARNING,
        REDUCED_DATA_PATH,
        REDUCED_DATA_NAME,
        REDUCTION_SIZE_FACTOR,
        INPUT_FILE_BASE_NAME,
        DATA_PATH,
        RAW_NAME,
        )


def load_predict(path=MODEL_PATH, version=VERSION, namePredictor=DEFAULT_PREDICTOR):
    """
    Simply load the predictions associated with the VERSION data
    """
    logging.info("trying to load {}".format(path + namePredictor + version + '.npz'))
    return np.load(path + namePredictor + version + '.npz')['pred']

def load_predict_byname(filename, path=MODEL_PATH):
    """
    Simply load the predictions associated with the VERSION data
    """
    full_path = os.path.join(path, filename)
    logging.info("trying to load {}".format(full_path))
    return np.load(os.path.join(path, filename))['pred']

def load_tSNE(params=PARAMS_LEARNING, version=VERSION, path=REDUCED_DATA_PATH,
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
            REDUCED_DATA_NAME,
            str(reduction_size_factor),
            name,
            version,
            '.npz',
        ])
        logging.info("embedded data= loading %s %s %s %s from %s",
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
            logging.info("embedded data=ready")
        else:
            logging.info("emebedded data = model {} {} {} {}  not found".format(
                perplexity,
                learning_rate,
                init,
                n_iter
                )
            )
    return x_transformed, models

def load_raw(version, path):
    raw_filename = os.path.join(path, RAW_NAME + version + '.npz')
    if os.path.exists(raw_filename):
        raw_ = np.load(raw_filename)
        return raw_["originals"], raw_["columns"]
    else:
        return None

def load_preprocessed(
        file_base_name=INPUT_FILE_BASE_NAME,
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
        - contains entry y with output data (possible_outputs_list to predict)
        - optionnaly an encoder to translate machine-readable possible_outputs_list to human-readable possible_outputs_list
                (actually it is the opposite e.g: {604600:[False, True, False, .., False]})

    :return: (input for t-SNE, classes, encoder (humantomachine class possible_outputs_list),
    decoder (machinetohuman possible_outputs_list))

    Note that encoder/decoder are assumed trivial if no encoder are found in npz

    """
    
    x_small = []
    y_small = []
    
    xy = np.load(path + INPUT_FILE_BASE_NAME + version + '.npz')
    
    """
    if output_name + '_encoder' in xy.keys():
        # I don't understant either but it is a 0-d array (...)
        class_encoder = xy[output_name + '_encoder'][()]
        decoder_dic = {
                class_encoder[k].argsort()[-1]: k
                for k in class_encoder.keys()
                }
        class_decoder = lambda oh: decoder_dic[np.argsort(oh)[-1]] # noqa
    else:
    """
    class_encoder = {y: y for y in set(y_small)}
    class_decoder = lambda x: x # noqa

    x = xy['x']
    x_small = x[:int(x.shape[0] / reduction_factor)]; del x # noqa
    if 'y' in xy.keys():
        y = xy['y']
        del xy
        return (np.array(x_small), np.array(y), class_encoder, class_decoder)
    elif 'y' + '_decoded' in xy.keys():
        y_decoded = xy['y' + '_decoded']
        del xy
        return (np.array(x_small), np.array(y_decoded), class_encoder, class_decoder)

