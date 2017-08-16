#!/usr/bin/python3
import logging

import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')  # noqa

from data_viz import dim_reduction
from data_viz import labelling
from data_viz import vizualization


from data_viz.config.references import (
    DATA_PATH,
    VERSION,
    REDUCTION_SIZE_FACTOR,
    TSNE_DATA_PATH,
    PARAMS,
    MODEL_PATH,
    DO_CALCULUS,
)

logging.basicConfig(level=logging.DEBUG)

logging.info("Starting script")

PARAMS['perplexities'] = [40, 50, 60, 70, 80]
PARAMS['learning_rates'] = [800, 1000]
PARAMS['inits'] = ['random', 'pca']
PARAMS['n_iters'] = [10000, 15000]
PARAM_VIZ = (80, 1000, 'random', 15000)

logging.info("raw_data=loading")
(
    x_small,
    y_small,
    class_encoder,
    class_decoder,

) = dim_reduction.load_raw_data()

logging.info('raw_data=loaded')

if DO_CALCULUS:
    logging.info("t-sne=learning")

    x_transformed, models = dim_reduction.learn_tSNE(
        PARAMS,
        VERSION,
        x_small,
        TSNE_DATA_PATH,
        REDUCTION_SIZE_FACTOR,
    )
    logging.info('t-sne=ready')
else:
    logging.info("t-sne=loading")

    x_transformed, models = dim_reduction.load_tSNE(
        PARAMS,
        VERSION,
        TSNE_DATA_PATH,
        REDUCTION_SIZE_FACTOR,
    )
    logging.info('t-sne=ready')

x_2D = x_transformed[PARAM_VIZ]

###############
# PREDICT

if DO_CALCULUS:
    logging.info('RNpredictions=predicting')
    x_predicted = labelling.predict_rnn(
        x_small,
        y_small,
        path=MODEL_PATH,
        version=VERSION
    )
    logging.info('RNpredictions=ready')
else:
    logging.info('RNpredictions=loading')
    x_predicted = labelling.load_predict(
        path=MODEL_PATH,
        version=VERSION
    )
    logging.info('RNpredictions=ready')
logging.info("loading raw transactions for analysis..")

transactions_raw = np.load(
    DATA_PATH + 'originals' + VERSION + '.npz'
)['originals']

f = vizualization.Vizualization(
    x_raw = transactions_raw,
    proj=x_2D,
    y_true=y_small,
    y_pred=x_predicted,
    resolution=200,
    class_decoder=class_decoder,
    class_encoder=class_encoder,
)

f.plot()
f.show()