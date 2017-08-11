"""
Query some datas directly from databases
Then prepare it for the vizualization
It should be launced only once in a while
"""
from config.references import (
        DATA_PATH,
        INPUT_FILE_BASE_NAME,
        RAW_NAME,
        MODEL_PATH,
        DEFAULT_PREDICTOR,
        VERSION,
        )
from config.manakin import (
        URI,
        )

import numpy as np
import ipdb
import logging
import os
from database_queries import db_interface

logging.basicConfig(level=logging.DEBUG)

def query_meta(DBI, set_name):
    meta_pk = DBI._register_algorithm_name("meta")
    oracle_pk = DBI._register_algorithm_name("final")
    transactions = DBI.get_training_inputs_auto_readable(set_name)
    transactions = [*transactions]
    
    return transactions, meta_pk, oracle_pk

def query_meta_all_sets(DBI):
    test, meta_pk, oracle_pk = query_meta(DBI, 'test')
    logging.info('query_test=ready\n')
    validation = query_meta(DBI, 'validation')[0]
    logging.info('query_validation=ready\n')
    test+= validation
    test = [ np.array(t) for t in test]
    test = np.array(test)
    logging.info('queries merged')


    return test, meta_pk, oracle_pk

def separate(datas, output_engine_pk, oracle_pk):
    raws = datas[:,range(len(datas[0])-1)]
    engines_results = datas[:,-1]

    inputs = [[] for _ in range(len(engines_results))]
    predictions=[0 for _ in range(len(engines_results))] # nothing predicted is class 0 predicted
    reality = [None for _ in range(len(engines_results))]

    for idx,r in enumerate(engines_results):
        for e in r:
            if e[0] == output_engine_pk:
                predictions[idx]= e
            elif e[0] == oracle_pk:
                reality[idx] = e[2]
            else:
                inputs[idx].append(e)


    return raws, inputs, predictions, reality

def preprocess_meta(raws, inputs, predictions,
        base_path=DATA_PATH,
        preprocessed_filename=INPUT_FILE_BASE_NAME,
        raw_filename=RAW_NAME,
        predictions_filename=DEFAULT_PREDICTOR,
        predictions_path=MODEL_PATH,
        version=VERSION,
        ):

    class_predicted = set()
    class_existing  = set()
    engines = set()
    
    # One-hot encoding for engines predictions feeding meta
    for input_ in inputs:
        for suggestion in input_:
            class_predicted.add(suggestion[2])
            engines.add(suggestion[0])
    engines = list(engines)

    for transaction in raws:
        class_existing.add(transaction[-3])
    for prediction in predictions:
        class_existing.add(prediction[2])

    all_class = class_predicted.union(class_existing)

    # builds encoder for the y
    encoder ={}
    encoding_compress = []
    blank_y = np.array([False]*len(all_class), dtype=bool)
    for idx, class_ in enumerate(all_class):
        encoding = blank_y.copy()
        encoding[idx] = True
        encoder[class_] = encoding
        encoding_compress.append(class_)

    # preprocess the x
    blank_x = np.array([0]*len(blank_y)*len(engines), dtype=float)
    xs=[]
    ys=[]

    for idx,transaction in enumerate(inputs):
        x = blank_x.copy()
        for prediction in transaction:
            engine_label = prediction[0]
            class_ = prediction[2]
            idx_vector = (engines.index(engine_label)*len(blank_y)
                     + encoding_compress.index(class_))
            x[idx_vector] = prediction[3]
        xs.append(x)
        ys.append(raws[idx][-3])

    if preprocessed_filename!='':
        np.savez(
                os.path.join(base_path,preprocessed_filename+version+'.npz'),
                x=xs,
                y_account_decoded=ys,
                account_encoder=encoder
                )
    if raw_filename!='':
        np.savez(
                os.path.join(base_path,raw_filename+version+'.npz'),
                originals=raws)
    if predictions_filename!='':
        np.savez(
                os.path.join(predictions_path, predictions_filename+version+'.npz'),
                pred=np.array(predictions)[:,2],
                )
    
    return xs, ys, encoder

def get_predictions_by_engine(
        datas,
        algo_names,
        get_algo_pk,
        model_path = MODEL_PATH,
        version=VERSION,
        save=True,
        ):
    
    pks=[]
    for algo_name in algo_names:
        pks.append(
                get_algo_pk(algo_name)
                )

    predictions_by_pk = {pk:[] for pk in pks}
    pk_has_proposed={pk:[] for pk in pks}

    for d in datas:
        pk_has_proposed={pk:[] for pk in pks}
        for prediction in d[-1]:
            pk = prediction[0]
            pk_has_proposed[pk].append(
                    (prediction[2],prediction[3])
                    )
        for pk, proposals in pk_has_proposed.items():
            if not proposals:
                predictions_by_pk[pk].append(0) # which means None
            else:
                predictions_by_pk[pk].append(
                        proposals[np.array(proposals)[:,1].argmax()][0]
                        )

    if save:
        for algo_name in algo_names:
            np.savez(
                    os.path.join(
                        model_path,
                        algo_name+'predict'+version+'.npz'),
                    pred=np.array(predictions)[:,2],
                    )
    return predictions_by_pk

if __name__=='__main__':

    MDI = db_interface.MetaDatabaseInterface(URI)
    get_algo_pk = MDI._register_algorithm_name

    logging.info('query db')
    datas, meta_pk, oracle_pk = query_meta_all_sets(MDI)
    logging.info('sort results')
    raws, inputs, predictions, reality = separate(datas, meta_pk, oracle_pk)
    logging.info('preprocess data')
    xs, ys, encoder = preprocess_meta(
            raws,
            inputs,
            predictions,
            )

    algo_names=['final', 'majika', 'tsuri', 'chiitoi', 'tango', 'meta']
    get_predictions_by_engine(datas, algo_names, get_algo_pk)
