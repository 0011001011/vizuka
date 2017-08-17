from data_viz.database_queries import db_interface
from data_viz.database_queries.query_and_process import (
        query_meta_all_sets,
        separate,
        preprocess_meta,
        get_predictions_by_engine,
        URI,
        )

DVDI = db_interface.DataVizDatabaseInterface(URI)
get_algo_pk = DVDI._register_algorithm_name

logger.info('query db')
datas, meta_pk, oracle_pk = query_meta_all_sets(DVDI)
logger.info('sort results')
raws, inputs, predictions, reality = separate(datas, meta_pk, oracle_pk)
logger.info('preprocess data')
xs, ys, encoder = preprocess_meta(
    raws,
    inputs,
    predictions,
)

algo_names = ['final', 'majika', 'tsuri', 'chiitoi', 'tango', 'meta']
get_predictions_by_engine(datas, algo_names, get_algo_pk)
