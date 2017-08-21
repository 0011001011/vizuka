from ml_shared import package_get_config, logger  # noqa

get_config = package_get_config(__package__)  # noqa

from data_viz.database_queries import db_interface
from data_viz.database_queries.query_and_process import (
        query_meta_all_sets,
        separate,
        preprocess_meta,
        get_predictions_by_engine,
        URI,
        make_translator,
        )

DVDI = db_interface.DataVizDatabaseInterface(URI)
get_algo_pk = DVDI._register_algorithm_name

datas, meta_pk, oracle_pk = query_meta_all_sets(DVDI)
raws, inputs, predictions, reality = separate(datas, meta_pk, oracle_pk)

translator = make_translator()

xs, ys, encoder = preprocess_meta(
    raws,
    inputs,
    predictions,
    translator=translator,
)

algo_names = ['final', 'majika', 'tsuri', 'chiitoi', 'tango', 'meta']
get_predictions_by_engine(datas, algo_names, get_algo_pk, translator=translator)
