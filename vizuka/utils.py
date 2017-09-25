"""
Useful functions such as a loader for the datas
"""
import numpy as np
from vizuka.config import (
    INPUT_FILE_BASE_NAME,
    REDUCTION_SIZE_FACTOR,
    VERSION,
    DATA_PATH,
    OUTPUT_NAME,
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
    
    xy = np.load(path + INPUT_FILE_BASE_NAME + version + '.npz')

    if output_name + '_encoder' in xy.keys():
        # I don't understant either but it is a 0-d array (...)
        class_encoder = xy[output_name + '_encoder'][()]
        decoder_dic = {
                class_encoder[k].argsort()[-1]: k
                for k in class_encoder.keys()
                }
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

