"""
This is the main script to launch everything
Is able to reduce the data and launch a vizualization
from it.
"""

import logging

import matplotlib
matplotlib.use('Qt5Agg')  # noqa
import numpy as np
import os
import argparse

from vizuka import dim_reduction
from vizuka import data_loader
from vizuka import vizualization


def main():

    """
    See --help if you want help
    """
    
    from vizuka.config import (
        DATA_PATH,
        VERSION,
        REDUCTION_SIZE_FACTOR,
        REDUCTED_DATA_PATH,
        PARAMS_LEARNING,
        PARAMS_VIZ,
        MODEL_PATH,
        INPUT_FILE_BASE_NAME,
        RAW_NAME,
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
        '--use_pca', 
         help='force a PCA dimensional reduction, needs a minimum variance ratio explained')
    parser.add_argument(
        '--version', type=str,
        help='(optional) specify a version of the files to load/generate, currently: '+VERSION)
    parser.add_argument(
        '--force_no_predict', action="store_true",
        help='do not load a predictions file : vizualize as if you predicted with 100% accuracy')
    parser.add_argument(
        '--no_vizualize', action="store_true",
         help='do not prepare a nice data vizualization')
    parser.add_argument(
        '--no_plot', action="store_true",
         help='do not show a nice data vizualization (but prepare it nonetheless)')
    parser.add_argument(
        '--path',
         help='(optional) location of your data/ folder, containing set/ reducted/ graph/ models/')
    parser.add_argument(
        '--mnist',
         help='use MNIST dataset')
    
    parser.set_defaults(
            no_plot=False, 
            no_vizualize=False,
            version=VERSION,
            path=os.path.dirname(__file__),
            feature_to_filter=[],
            feature_to_show=[],
            use_pca = 0,
            force_no_predict = False,
            mnist=False,
            )

    args = parser.parse_args()

    MODEL_PATH = os.path.join(args.path, MODEL_PATH)
    INPUT_FILE_BASE_NAME = os.path.join(args.path, INPUT_FILE_BASE_NAME)
    REDUCTED_DATA_PATH = os.path.join(args.path, REDUCTED_DATA_PATH)
    DATA_PATH = os.path.join(args.path, DATA_PATH)

    reduce_      = args.reduce
    no_vizualize = args.no_vizualize
    no_plot      = args.no_plot
    version      = args.version
    features_name_to_filter  = args.feature_to_filter
    features_name_to_display = args.feature_to_show
    pca_variance_needed = args.use_pca
    force_no_predict = args.force_no_predict
    
    if args.mnist:
        from example import load_mnist
        version = load_mnist.version
    
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
        x,
        y,
        class_encoder,
        class_decoder,
    ) = data_loader.load_preprocessed(
            file_base_name   = INPUT_FILE_BASE_NAME,
            path             = DATA_PATH,
            version          = version,
            reduction_factor = REDUCTION_SIZE_FACTOR
            )

    logging.info('raw_data=loaded')

    x_transformed = {}
    if not reduce_:
        logging.info("t-sne=loading")

        x_transformed, models = data_loader.load_tSNE(
            params                = PARAMS_LEARNING,
            version               = version,
            path                  = REDUCTED_DATA_PATH,
            reduction_size_factor = REDUCTION_SIZE_FACTOR,
        )
        logging.info("found version:{} with"
                     "{} different sets of reducted data".format(
                            version, len(x_transformed)
                            )
                     )

        logging.info('t-sne=ready')

    if not x_transformed:
        logging.info("no reduced data found! Needs to learn some dimension reduction..")
        force_reduce = True
    else:
        force_reduce = False

    if force_reduce or reduce_: # if nothing loaded or reduce is forced by arg
        logging.info("t-sne=learning")

        x_transformed, models = dim_reduction.learn_tSNE(
            x                       = x,
            params                  = PARAMS_LEARNING,
            version                 = version,
            path                    = REDUCTED_DATA_PATH,
            reduction_size_factor   = REDUCTION_SIZE_FACTOR,
            pca_variance_needed     = pca_variance_needed,
        )

        logging.info('t-sne=ready')
    
    param_to_vizualize = (
            PARAMS_VIZ['perplexity'],
            PARAMS_VIZ['learning_rate'],
            PARAMS_VIZ['init'],
            PARAMS_VIZ['n_iter'],
            )

    if param_to_vizualize in x_transformed.keys():
        x_2D = x_transformed[param_to_vizualize]
    else:
        random_param_to_load = list(x_transformed.keys())[0]
        logging.info("PARAM_VIZ specified in config.py not found, but other data exists"
                     " will load {} instead (chosen randomly)".format(random_param_to_load)
                     )
        x_2D = x_transformed[random_param_to_load]


    ###############
    # PREDICT
    if force_no_predict:
        x_predicted = y
    else:
        try:
            logging.info('predictions=loading')
            x_predicted = data_loader.load_predict(
                path=MODEL_PATH,
                version=version,
            )
            logging.info('RNpredictions=ready')
        except FileNotFoundError:
            logging.info((
                    "Nothing found in {}, no predictions to vizualize\n"
                    "if this is intended you can force the vizualization"
                    "with --force_no_predict :\n{}\n"
                    ).format(MODEL_PATH, os.listdir(MODEL_PATH)))
            return
    
    raw = data_loader.load_raw(version, DATA_PATH)
    if raw:
        logging.info("loading raw transactions for analysis..")
        raw_data, raw_columns = raw
    else:
        logging.info('no raw data provided, all cluster vizualization disabled! (-s and -f options)')
        raw_data    = []
        raw_columns = []
        features_name_to_filter  = []
        features_name_to_display = {}


    
    if not no_vizualize:

        f = vizualization.Vizualization(
            projected_input=x_2D,
            predicted_outputs=x_predicted,
            raw_inputs=raw_data,
            raw_inputs_columns=raw_columns,
            correct_outputs=y,
            resolution=200,
            class_decoder=class_decoder,
            class_encoder=class_encoder,
            special_class='x',
            nb_of_clusters=120,
            features_name_to_filter  = features_name_to_filter,
            features_name_to_display = features_name_to_display,
            output_path=os.path.join('output.csv'),
            model_path=MODEL_PATH,
            version=version,
        )

        if not no_plot:
            f.plot()
            f.show()
    return f

if __name__ == '__main__':
    main()
