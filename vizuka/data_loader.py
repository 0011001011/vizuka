import logging
import os
import itertools

import numpy as np

from vizuka import dimension_reduction
from vizuka.dimension_reduction.projector import Projector
from vizuka.config import (
        MODEL_PATH,
        VERSION,
        DEFAULT_PREDICTOR,
        PROJECTION_DEFAULT_PARAMS,
        REDUCED_DATA_PATH,
        REDUCED_DATA_NAME,
        INPUT_FILE_BASE_NAME,
        DATA_PATH,
        RAW_NAME,
        )


def load_predict(path=MODEL_PATH, version=VERSION, namePredictor=DEFAULT_PREDICTOR):
    """
    Simply load the predictions associated with the VERSION data
    """
    logging.info("trying to load {}".format(path + namePredictor +'_'+ version + '.npz'))
    return np.load(path + namePredictor + version + '.npz')['pred']

def load_predict_byname(filename, path=MODEL_PATH):
    """
    Simply load the predictions associated with the VERSION data
    """
    full_path = os.path.join(path, filename)
    logging.info("trying to load {}".format(full_path))
    return np.load(os.path.join(path, filename))['pred']

def list_projections(reduced_path):
    """
    Returns a list containing [(algo_name, parameters),..] found in :param reduced_path:
    """
    files = [filename for filename in os.listdir(reduced_path) if ".npz" in filename]
    return  [Projector.get_param_from_name(filename[:-4]) for filename in files]

def load_projection(algorithm_name, parameters, version, path):
    logging.info("data_loader=loading projection:\n\talgorithm:{}\n\tparameters:{}".format(
                        algorithm_name,
                        parameters,
                        ))
    algo_builder = dimension_reduction.make_projector(algorithm_name)
    algo = algo_builder(**parameters)
    projection = algo.load_projection(version=version, path=path)

    logging.info("data_loader=ready")
    return projection

def load_raw(version, path):
    raw_filename = os.path.join(path, RAW_NAME + version + '.npz')
    if os.path.exists(raw_filename):
        raw_ = np.load(raw_filename)
        return raw_["x"], raw_["columns"]
    else:
        return None

def load_preprocessed(
        file_base_name=INPUT_FILE_BASE_NAME,
        path=DATA_PATH,
        version=VERSION,
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
    xy = np.load(path + INPUT_FILE_BASE_NAME + version + '.npz')
    x, y = xy['x'], xy['y']
    
    class_encoder = {y: y for y in set(y)}
    class_decoder = lambda x: x # noqa

    x = xy['x']
    if 'y' in xy.keys():
        y = xy['y']
        del xy
        return (np.array(x), np.array(y), class_encoder, class_decoder)
    elif 'y' + '_decoded' in xy.keys():
        y_decoded = xy['y' + '_decoded']
        del xy
        return (np.array(x), np.array(y_decoded), class_encoder, class_decoder)

