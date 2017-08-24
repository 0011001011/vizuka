"""
This is the main script to launch everything
Is able to reduce the data and launch a vizualization
from it.
"""

import logging

import matplotlib
import numpy as np
import os

matplotlib.use('Qt5Agg')  # noqa

from data_viz import dim_reduction
from data_viz import labelling
from data_viz import vizualization


import argparse

def main():

    """
    See --help if you want help
    """
    
    from data_viz.config import (
        DATA_PATH,
        VERSION,
        REDUCTION_SIZE_FACTOR,
        TSNE_DATA_PATH,
        PARAMS_LEARNING,
        PARAMS_VIZ,
        MODEL_PATH,
        INPUT_FILE_BASE_NAME,
        OUTPUT_NAME,
    )


    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--reduce', action="store_true",
         help='launch a full dimension reduction')
    parser.add_argument(
        '--version', type=str,
        help='specify a version of the files to load/generate, currently: '+VERSION)
    parser.add_argument(
        '--do_vizualize', action="store_false",
         help='do not prepare a nice data vizualization')
    parser.add_argument(
        '--do_plot', action="store_false",
         help='do not show a nice data vizualization (but prepare it nonetheless)')
    
    parser.set_defaults(do_plot=True, do_vizualize=True, version=VERSION)

    args = parser.parse_args()

    reduce_      = args.reduce
    do_vizualize = args.do_vizualize
    do_plot      = args.do_plot
    version      = args.version

    logging.info("Starting script")
    logging.info("raw_data=loading")
    (
        x_small,
        y_small,
        class_encoder,
        class_decoder,

    ) = dim_reduction.load_raw_data(
            file_base_name   = INPUT_FILE_BASE_NAME,
            output_name      = OUTPUT_NAME,
            path             = DATA_PATH,
            version          = VERSION,
            reduction_factor = REDUCTION_SIZE_FACTOR
            )

    logging.info('raw_data=loaded')

    if reduce_:
        logging.info("t-sne=learning")

        x_transformed, models = dim_reduction.learn_tSNE(
            x                       = x_small,
            params                  = PARAMS_LEARNING,
            version                 = VERSION,
            path                    = TSNE_DATA_PATH,
            reduction_size_factor   = REDUCTION_SIZE_FACTOR,
            pca_components          = None,
        )

        logging.info('t-sne=ready')
    else:
        logging.info("t-sne=loading")

        x_transformed, models = dim_reduction.load_tSNE(
            params                = PARAMS_LEARNING,
            version               = VERSION,
            path                  = TSNE_DATA_PATH,
            reduction_size_factor = REDUCTION_SIZE_FACTOR,
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

    if do_vizualize:

        f = vizualization.Vizualization(
            raw_inputs=transactions_raw,
            projected_input=x_2D,
            predicted_outputs=x_predicted,
            correct_outputs=y_small,
            resolution=200,
            class_decoder=class_decoder,
            class_encoder=class_encoder,
            special_class='0',
            number_of_clusters=120
            output_path   = os.path.join(os.path.__file__, 'output.csv'),
            model_path    = MODEL_PATH,
        )

        if do_plot:
            f.plot()
            f.show()

if __name__ == '__main__':
    main()
