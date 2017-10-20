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

from vizuka import dimension_reduction
from vizuka import data_loader
from vizuka import vizualization
from vizuka import launch_reduce


def main():

    """
    See --help if you want help
    """
    
    from vizuka.config import (
        BASE_PATH,
        DATA_PATH,
        VERSION,
        REDUCED_DATA_PATH,
        REDUCED_DATA_NAME,
        MODEL_PATH,
        INPUT_FILE_BASE_NAME,
        RAW_NAME,
        PROJECTION_DEFAULT_PARAMS,
        DEFAULT_PROJECTOR,
    )


    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mnist', action='store_true',
         help='download, fit, and vizualize MNIST dataset')
    parser.add_argument(
            '--show-required-files', action='store_true',
            help='show all the required files for optimal use',
            )
    parser.add_argument(
        '-p', '--path',
         help='change the location of your data/ folder, containing set/ reducted/ graph/ models/')
    parser.add_argument(
        '-f', '--feature_to_filter', action='append',
         help='Adds a feature listed in raw_data.npz["columns"] to the filters list')
    parser.add_argument(
        '-s', '--feature_to_show', action='append',
        help='Usage : -s MY_COLUMN_NAME:PLOTTER with PLOTTER being a value in %. Adds this non-preprocessed/human-readable feature to the cluster view'.format())
    parser.add_argument(
        '-r', '--reduce', action="store_true",
         help='launch a full dimension reduction')
    parser.add_argument(
        '-h1', '--heatmap1',
         help='Specify the 1st heatmap to show')
    parser.add_argument(
        '-h2', '--heatmap2',
         help='Specify the 2nd heatmap to show')
    parser.add_argument(
        '--use_pca', 
         help='force a PCA dimensional reduction, needs a minimum variance ratio explained')
    parser.add_argument(
        '-v', '--version', type=str,
        help='(optional) specify a version of the files to load/generate (see vizuka --show_required_files), currently: '+VERSION)
    parser.add_argument(
        '--force-no-predict', action="store_true",
        help='(not recommended) do not load a predictions file : vizualize as if you predicted with 100\% accuracy')
    parser.add_argument(
        '--no-vizualize', action="store_true",
         help='(for debug) do not prepare a nice data vizualization')
    parser.add_argument(
        '--no-plot', action="store_true",
         help='(for debug) do not show a nice data vizualization (but prepare it nonetheless)')
    
    parser.set_defaults(
            heatmap1='accuracy',
            heatmap2='entropy',
            no_plot=False, 
            no_vizualize=False,
            show_required_files=False,
            version=VERSION,
            path=os.path.join(os.path.dirname(__file__),BASE_PATH),
            feature_to_filter=[],
            feature_to_show=[],
            use_pca = 0,
            force_no_predict = False,
            mnist=False,
            )

    args = parser.parse_args()
    
    path                 = args.path
    MODEL_PATH           = os.path.join(args.path, MODEL_PATH)
    INPUT_FILE_BASE_NAME = os.path.join(args.path, INPUT_FILE_BASE_NAME)
    REDUCED_DATA_PATH    = os.path.join(args.path, REDUCED_DATA_PATH)
    DATA_PATH            = os.path.join(args.path, DATA_PATH)

    reduce_      = args.reduce
    no_vizualize = args.no_vizualize
    no_plot      = args.no_plot
    version      = args.version
    features_name_to_filter  = args.feature_to_filter
    features_name_to_display = args.feature_to_show
    pca_variance_needed = args.use_pca
    force_no_predict = args.force_no_predict
    show_required_files=args.show_required_files
    heatmap1    = args.heatmap1
    heatmap2    = args.heatmap2

    if args.show_required_files:
        print(
        'VERSION: string that identifies your dataset (default is MNIST_example)\n\n'
        '\nVizuka needs the following files :\n\n'
        '\t + data/set/preprocessed_inputs_VERSION.npz\n'
        '\t ------------------------------------------\n'
        '\t\t x:\t preprocessed inputs\n'
        '\t\t y:\t outputs to be predicted\n'
        '\t\t NB:\t this is the only mandatory file, the following is highly recommended:\n'
        '\n\n'
        '\t + data/models/predict_VERSION.npz -> optional but recommended\n'
        '\t -------------------------------------------------------------\n'
        '\t\t pred:\t predictions returned by your algorithm\n'
        '\t\t NB:\t should be same formatting as in preprocessed_inputs_VERSION["y"])\n'
        '\n\n'
        '\t + raw_data.npz -> optional\n'
        '\t --------------------------\n'
        '\t\t x:\t\t array of inputs BEFORE preprocessing\n'
        '\t\t\t\t\t probably human-readbable, thus useful for vizualization\n'
        '\t\t columns:\t the name of the columns variable in x\n'
        '\t\t NB:\t this file is used if you run vizuka with\n'
        '\t\t\t    --feature-name-to-display COLUMN_NAME:PLOTTER COLUMN_NAME2:PLOTTER2 or\n'
        '\t\t\t    --feature-name-to-filter COLUMN_NAME1 COLUMN_NAME2 (see help for details)\n'
        '\n\n'
        '\t + reduced/2Dembedding_PARAMS_VERSION.npz -> reaaaally optional\n'
        '\t --------------------------------------------------------------\n'
        '\t\t x2D:\t projections of the preprocessed inputs x in a 2D space\n'
        '\t\t NB:\t this set is automatically generated with tSNE but you can specify your own\n'
        )
        return

    
    if args.mnist:
        from vizuka.example import load_mnist
        version = load_mnist.version
        features_name_to_display = ['image:images']

    
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
            )

    logging.info('raw_data=loaded')

    x_2D = []
    if not reduce_:
        x_2D = data_loader.load_projection(
            algorithm_name        = DEFAULT_PROJECTOR,
            parameters            = PROJECTION_DEFAULT_PARAMS[DEFAULT_PROJECTOR],
            version               = version,
            path                  = REDUCED_DATA_PATH,
            base_filename         = REDUCED_DATA_NAME,
        )

    if not x_2D.size:
        logging.info("no reduced data found! Needs to learn some dimension reduction..")
        force_reduce = True
    else:
        force_reduce = False

    if force_reduce or reduce_: # if nothing loaded or reduce is forced by arg

        x_2D = launch_reduce.do_reduce(
            algorithm_name          = DEFAULT_PROJECTOR,
            parameters              = PROJECTION_DEFAULT_PARAMS[DEFAULT_PROJECTOR],
            version                 = version,
            data_path               = DATA_PATH,
            reduced_path            = REDUCED_DATA_PATH,
        )
    
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
            heatmaps_requested = [heatmap1, heatmap2],
            output_path=os.path.join('output.csv'),
            base_path=path,
            version=version,
        )

        if not no_plot:
            f.plot()
            f.show()
    return f

if __name__ == '__main__':
    main()
