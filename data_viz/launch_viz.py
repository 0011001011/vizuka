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

import dim_reduction
import labelling
import vizualization


import argparse

def main():

    """
    See --help if you want help
    """
    
    from config import (
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
        '-f', '--feature_to_filter', action='append',
         help='Adds a feature listed in originals.npz["columns"] to the filters list')
    parser.add_argument(
        '-s', '--feature_to_show', action='append',
         help='Adds a feature listed in originals.npz["columns"] to the cluster view')
    parser.add_argument(
        '--reduce', action="store_true",
         help='launch a full dimension reduction')
    parser.add_argument(
        '--version', type=str,
        help='(optional) specify a version of the files to load/generate, currently: '+VERSION)
    parser.add_argument(
        '--no_vizualize', action="store_true",
         help='do not prepare a nice data vizualization')
    parser.add_argument(
        '--no_plot', action="store_true",
         help='do not show a nice data vizualization (but prepare it nonetheless)')
    parser.add_argument(
        '--path',
         help='(optional) location of your data/ folder, containing set/ tSNE/ graph/ models/')
    
    parser.set_defaults(
            no_plot=False, 
            no_vizualize=False,
            version=VERSION,
            path=os.path.dirname(__file__),
            feature_to_filter=[],
            feature_to_show=[],
            )

    args = parser.parse_args()

    MODEL_PATH = os.path.join(args.path, MODEL_PATH)
    INPUT_FILE_BASE_NAME = os.path.join(args.path, INPUT_FILE_BASE_NAME)
    TSNE_DATA_PATH = os.path.join(args.path, TSNE_DATA_PATH)
    DATA_PATH = os.path.join(args.path, DATA_PATH)


    reduce_      = args.reduce
    no_vizualize = args.no_vizualize
    no_plot      = args.no_plot
    version      = args.version
    features_name_to_filter  = args.feature_to_filter
    features_name_to_display = args.feature_to_show
    
    new_fntd = {}
    error_msg  = 'Argument should be feature_name:plotter with plotter'
    error_msg += '\nin (logdensity, density, wordcloud, counter)'
    e = Exception(error_msg)
    for feature_name in features_name_to_display:
        if ':' not in feature_name:
            raise e
        k,v = feature_name.split(':')
        # TODO if v not in cluster_diving.plotter.keys():
        #    raise e
        plotters = new_fntd.get(k, [])
        plotters.append(v)
        new_fntd[k] = plotters
    features_name_to_display = new_fntd

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
    transactions_raw_ = np.load(
        DATA_PATH + 'originals' + VERSION + '.npz'
    )
    
    data_unique_id_string = '_filesep_'.join([str(PARAMS_VIZ['perplexity']),
                                              str(PARAMS_VIZ['learning_rate']),
                                              str(PARAMS_VIZ['init']),
                                              str(PARAMS_VIZ['n_iter']),
                                              str(VERSION),
                                              str(TSNE_DATA_PATH).replace('/', '_')
                                             ])
    transactions_raw = transactions_raw_['originals']
    transactions_columns = transactions_raw_['columns']

    if not no_vizualize:

        f = vizualization.Vizualization(
            raw_inputs=transactions_raw,
            raw_inputs_columns=transactions_columns,
            projected_input=x_2D,
            predicted_outputs=x_predicted,
            correct_outputs=y_small,
            resolution=200,
            class_decoder=class_decoder,
            class_encoder=class_encoder,
            special_class='0',
            number_of_clusters=120,
            features_name_to_filter  = features_name_to_filter,
            features_name_to_display = features_name_to_display,
            output_path=os.path.join(os.path.__file__, 'output.csv'),
            model_path=MODEL_PATH,
            data_unique_id_string=data_unique_id_string
        )

        if not no_plot:
            f.plot()
            f.show()

if __name__ == '__main__':
    main()
