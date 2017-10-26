"""
This is the main script to launch everything
Is able to reduce the data and launch a vizualization
from it.
"""

import logging
import os
import argparse

import matplotlib
matplotlib.use('Qt5Agg')  # noqa
import numpy as np
from pyfiglet import Figlet

from vizuka import data_loader
from vizuka import vizualization
from vizuka import launch_reduce

logger = logging.getLogger()
logger.setLevel(logging.WARN)

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
        PROJECTION_DEFAULT_PARAMS,
        DEFAULT_PROJECTOR,
    )

    print(Figlet().renderText('Vizuka'))

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
    parser.add_argument(
            '--verbose', action="store_true",
            help="verbose mode")
    
    parser.set_defaults(
            heatmap1 ='accuracy',
            heatmap2 ='entropy',
            use_pca  = 0,
            no_plot             =False,
            no_vizualize        =False,
            show_required_files =False,
            force_no_predict    =False,
            mnist               =False,
            version =VERSION,
            path    =os.path.join(os.path.dirname(__file__),BASE_PATH),
            feature_to_filter =[],
            feature_to_show   =[],
            verbose = False,
            )

    args = parser.parse_args()
    
    path                 = args.path
    MODEL_PATH           = os.path.join(args.path, MODEL_PATH)
    INPUT_FILE_BASE_NAME = os.path.join(args.path, INPUT_FILE_BASE_NAME)
    REDUCED_DATA_PATH    = os.path.join(args.path, REDUCED_DATA_PATH)
    DATA_PATH            = os.path.join(args.path, DATA_PATH)

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
    verbose     = args.verbose

    if verbose:
        logger.setLevel(logging.DEBUG)

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

    logger.info("Starting script")
    logger.info("raw_data=loading")
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

    logger.info('raw_data=loaded')

    projections_available = data_loader.list_projections(REDUCED_DATA_PATH)

    if len(projections_available)==0:
        logger.warn("No reduced data found! Please use vizuka-reduce to generate some")
        return

    choice_list, choice_dict = "", {}

    for i,(method, version_, params), in enumerate(projections_available):
        param_str = ''.join(["\t\t\t{}: {}\n".format(name, value) for name, value in params.items()])
        choice_list+="\t [{}]: \t{}\n\t\tparameters:\n{}\n".format(i, method, param_str)
        choice_dict[i]=method
    choice = input( "Projections available: (generate more with vizuka-reduce)"
                    ")\n"+choice_list+"\t[?] > ")
    try:
        choice_int=int(choice)
    except:
        logging.warn("Please enter a valid integer !\nABORTING")
        return
    
    selected_method      = choice_dict[choice_int]
    selected_projections, selected_version, selected_params = projections_available[choice_int]

    if selected_version != version:
        logging.warn(
                "Mismatch between file VERSION ({}) and requested VERSION ({})".format(
                selected_version, version))
        logging.warn("See --version\nABORTING")
        return

    x_2D = data_loader.load_projection(
        algorithm_name        =   selected_method,
        parameters            =   selected_params,
        version               =           version,
        path                  = REDUCED_DATA_PATH,
    )
    
    ###############
    # PREDICT
    if force_no_predict:
        x_predicted = y
    else:
        try:
            logger.info('predictions=loading')
            x_predicted = data_loader.load_predict(
                path=MODEL_PATH,
                version=version,
            )
            logger.info('RNpredictions=ready')
        except FileNotFoundError:
            logger.info((
                    "Nothing found in {}, no predictions to vizualize\n"
                    "if this is intended you can force the vizualization"
                    "with --force_no_predict :\n{}\n"
                    ).format(MODEL_PATH, os.listdir(MODEL_PATH)))
            return
    
    raw = data_loader.load_raw(version, DATA_PATH)
    if raw:
        logger.info("loading raw transactions for analysis..")
        raw_data, raw_columns = raw
    else:
        logger.info('no raw data provided, all cluster vizualization disabled! (-s and -f options)')
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
