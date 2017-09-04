"""
Functions needed to predict/loadpredictions/format labels
Not interesting to read
"""

import numpy as np

from vizuka.config import (
    MODEL_PATH,
    VERSION,
    DEFAULT_PREDICTOR,
)

class Predictor():
    def predict(x, **kwargs):
        pass


class MetaPredict():
    def __init__(self, ordered_predictions):
        self.predictions = ordered_predictions
        
    def predict(self, xs):
        return [self.predictions[self.predictions.indexof(x)] for x in xs]


def predict_rnn(
    x, y,
    path=MODEL_PATH,
    version=VERSION,
    nameRN=DEFAULT_PREDICTOR,
    format_from_one_hot=True,
    save=True,
):

    """
    Run the predict function on (:param x:,:param y:)
    Will be using a RN, which may need to reformulate its prediction
        ([0.001, 0.98767,0.00001678] should become [0,1,0])

    :param path: path to look for the models
    :param version: version of the model (e.g 20170614)
    :param format_from_one_hot: boolean to transform (0.00000461,0.00044486,0.99984] to [0,0,1]
    :param save: if True predictions will be saved in :param path:

    :return: vector of predictions
    """
    import keras
    predictor = keras.models.load_model(path + nameRN + version)
    x_predicted = predictor.predict(x)

    # x_predicted_one_hot needs to be reformulated a bit
    if format_from_one_hot:
        x_predicted = reformat_predict(x_predicted)

    x_predicted = np.array(x_predicted)
    if save:
        np.savez(
            path + 'predictions' + version + '.npz',
            pred=x_predicted
        )

    return x_predicted


def load_predict(path=MODEL_PATH, version=VERSION, namePredictor=DEFAULT_PREDICTOR):
    """
    Simply load the predictions stored with predict_rnn function
    """
    return np.load(path + namePredictor + version + '.npz')['pred']


def reformat_predict(predictions):
    """
    Return [0,0,1] from [0.00009493, 0.000036783,0.99345]
    """

    blank = [False] * len(predictions[0])
    x_predicted_formatted = []

    for i in predictions:
        one_hot_vector = blank.copy()
        one_hot_vector[i.argsort()[-1]] = True
        x_predicted_formatted.append(one_hot_vector)

    return x_predicted_formatted
