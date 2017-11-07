import argparse
import logging
from pyfiglet import Figlet

from vizuka import dimension_reduction
from vizuka import data_loader
from vizuka import config

from vizuka.config import (
    VERSION,
    INPUT_FILE_BASE_NAME,
    DATA_PATH,
    BASE_PATH,
    PROJECTION_DEFAULT_PARAMS,
)

logger = logging.getLogger()
logger.setLevel(logging.WARN)

def do_reduce(algorithm_name, parameters, version, data_path, reduced_path):
    """
    Project the data in :param data_path: (version :param version:)
    with algorithm :param algorithm_name: using parameters in :param parameters
    and save it in :param reduced_path:
    """
    algo_builder = dimension_reduction.make_projector(algorithm_name)
    algo = algo_builder(**parameters)

    (x, _ ,_ ,_, loaded, preprocessed_filename ) = data_loader.load_preprocessed(
            file_base_name   = INPUT_FILE_BASE_NAME,
            path             = data_path,
            version          = version,
            )
    if not loaded:
        logging.warn("\nNo data found\nCorresponding file not found: {}\nPlease check --show-required-files".format(preprocessed_filename))
        return
    
    logging.warn("Projecting the preprocessed data in 2D.. this may take a while!")
    algo.project(x)                                   # do the dimension projection
    algo.save_projection(version=version, path=reduced_path)     # save the result
    logging.warn("Projecting done")


def main():
    """
    Loads parameters and run do_reduce
    """

    print(Figlet().renderText('Vizuka'))

    builtin_projectors, extra_projectors = dimension_reduction.list_projectors()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--algorithm',
        help=(
            'algorithm name to reduce dimensions, available are :\n'
             + 'builtin: {}\n'.format(list(builtin_projectors.keys()))
             + 'extra plugins: {}'.format(list(extra_projectors.keys()))
             )
         )
    parser.add_argument(
        '-p', '--parameters', action='append',
        help='specify parameters, e.g: "-p perplexity:50 -p learning_rate:0.1" '
             'It will load default values in config.py if not specified'
        )
    parser.add_argument(
        '-v', '--version',
        help='specify a version of the files to load/generate (see vizuka --show_required_files), currently: '+VERSION)
    parser.add_argument(
        '--path',
        help='change the location of your data/ folder containing reduced/ (the projections)'
             ' and set/ (the raw and preprocessed data).'
             ' Default to {}'.format(BASE_PATH)
        )
    parser.add_argument(
            '--verbose', action="store_true",
            help="verbose mode")

    parser.set_defaults(
            algorithm = 'manual',
            parameters  = {},
            path        = BASE_PATH,
            version     = VERSION,
            )

    args = parser.parse_args()

    (data_path, reduced_path, _, _, _, _) = config.path_builder(args.path)

    algorithm_name = args.algorithm
    parameters     = args.parameters
    version        = args.version
    verbose        = args.verbose

    print("VERSION: Loading dataset labeled {} (cf --version)".format(version))
    
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    if algorithm_name == 'manual':
        choice_list, choice_dict = "", {}
        for i,method in enumerate([*builtin_projectors, *extra_projectors]):
            choice_list+="\t\t [{}]: {}\n".format(i, method)
            choice_dict[i]=method
        choice = input("No algorithm specified (-a or --algorithm)\n\tChoices available are:\n"+choice_list+"\t[?] > ")
        try:
            choice_int=int(choice)
        except:
            logger.warn("Please enter a valid integer !")
            return
        algorithm_name = choice_dict[choice_int]
              

    for raw_param in parameters:
        if ':' not in raw_param:
            raise TypeError('parameter -p not used correctly! see --help')
        param_name, param_value = raw_param.split(':')
        parameters[param_name:param_value]

    default_params = PROJECTION_DEFAULT_PARAMS.get(algorithm_name, {})
    for default_param_name, default_param_value in default_params.items():
        if default_param_name not in parameters:
            parameters[default_param_name] = default_param_value
            logger.info('parameter {} not specified, using default value: {}'.format(
                                default_param_name, default_param_value)
                            )

    do_reduce(algorithm_name, parameters, version, data_path, reduced_path)
    
    cmd_to_launch = "vizuka"
    cmd_to_launch+= (' --version {}'.format(version) if version!=VERSION else '')
    cmd_to_launch+= ('--path {}'.format(args.path) if args.path != BASE_PATH else '')
    print('"Projection done, you can now launch "vizuka"')

if __name__=='__main__':
    main()
