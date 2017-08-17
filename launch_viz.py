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
    PARAMS_LEARNING,
    PARAMS_VIZ,
    MODEL_PATH,
    DO_CALCULUS,
)

import argparse

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--reduce', action="store_true",
         help='launch a full dimension reduction')
    parser.add_argument(
        '--version', type=str,
        help='specify a version of the files to load/generate, currently: '+VERSION)
    parser.add_argument(
        '--no_vizualize', action="store_true",
         help='do not show a nice data vizualization (but prepare it nonetheless)')

    args = parser.parse_args()

    if args.version:
        VERSION=args.version
        logging.info("Overriding VERSION")
    
    logging.info("Starting script")
    logging.info("raw_data=loading")
    (
        x_small,
        y_small,
        class_encoder,
        class_decoder,

    ) = dim_reduction.load_raw_data()

    logging.info('raw_data=loaded')

    if args.reduce:
        logging.info("t-sne=learning")

        x_transformed, models = dim_reduction.learn_tSNE(
            x = x_small,
            params = PARAMS_LEARNING,
            version = VERSION,
            path = TSNE_DATA_PATH,
            reduction_size_factor = REDUCTION_SIZE_FACTOR,
        )
        logging.info('t-sne=ready')
    else:
        logging.info("t-sne=loading")

        x_transformed, models = dim_reduction.load_tSNE(
            PARAMS_LEARNING,
            VERSION,
            TSNE_DATA_PATH,
            REDUCTION_SIZE_FACTOR,
        )
        logging.info('t-sne=ready')

    x_2D = x_transformed[
            PARAMS_VIZ['perplexity'],
            PARAMS_VIZ['learning_rate'],
            PARAMS_VIZ['init'],
            PARAMS_VIZ['n_iter'],
            ]

    ###############
    # PREDICT
    
    """
    if DO_CALCULUS:
        logging.info('RNpredictions=predicting')
        x_predicted = labelling.predict_rnn(
            x_small,
            y_small,
            path=MODEL_PATH,
            version=VERSION
        )
        logging.info('RNpredictions=ready')
    """
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

    if not args.no_vizualize:

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
