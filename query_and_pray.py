"""
Query some datas directly from databases
Then prepare it for the vizualization
It should be launced only once in a while
"""
import numpy as np
import ipdb
import logging
from database_query import db_interface

uri = 'postgres://ds:ds@localhost/gold_standard_manakin'
logging.basicConfig(level=logging.DEBUG)

def query_meta(uri, set_name):
    MDI = db_interface.MetaDatabaseInterface(uri)
    meta_name = MDI._register_algorithm_name("meta")
    transactions = MDI.get_training_inputs_auto_readable(set_name)
    transactions = [*transactions]
    
    return transactions, meta_name

def query_meta_all_sets(uri):
    test = query_meta(uri, 'test')
    logging.info('query_test=ready\n')
    meta_pk = test[1]
    test=test[0]
    validation = query_meta(uri, 'validation')[0]
    logging.info('query_validation=ready\n')
    test+= validation
    test = [ np.array(t) for t in test]
    test = np.array(test)
    logging.info('queries merged')


    return test, meta_pk

def separate(datas, output_engine_pk):
    raws = datas[:,range(len(datas[0])-1)]
    engines_results = datas[:,-1]

    inputs = [[] for _ in range(len(engines_results))]
    predictions=[None for _ in range(len(engines_results))]
    reality = [None for _ in range(len(engines_results))]
    oracle_pk = 3

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
        name_file="xy.npz",
        name_originals='originals.npz',
        name_predictions='predictions.npz',
        save=False):

    class_predicted = set()
    class_existing  = set()
    engines = set()

    for input_ in inputs:
        for suggestion in input_:
            class_predicted.add(suggestion[2])
            engines.add(suggestion[0])
    engines = list(engines)

    for transaction in raws:
        class_existing.add(transaction[-4])
    for transaction in predictions:
        class_existing.add(transaction[-4])

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

    if save:
        np.savez(name_file, x=xs, y_account_decoded=ys, account_encoder=encoder)
        np.savez(name_originals, originals=raws)
        np.savez(name_predictions, pred=predictions)
    
    return xs, ys, encoder

if __name__=='__main__':
    logging.info('query db')
    datas, meta_pk = query_meta_all_sets(uri)
    logging.info('sort results')
    raws, inputs, predictions, reality = separate(datas, meta_pk)
    logging.info('preprocess data')
    xs, ys, encoder = preprocess_meta(raws, inputs, predictions, save=True)
