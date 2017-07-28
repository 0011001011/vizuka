"""
Query some datas directly from databases
Then prepare it for the vizualization
It should be launced only once in a while
"""
import numpy as np
import ipdb

uri = 'postgres://ds:ds@localhost/gold_standard_manakin'

def query_meta(uri, set_name):
    import db_interface
    MDI = db_interface.MetaDatabaseInterface(uri)
    meta_name = MDI._register_algorithm_name("meta")
    transactions = MDI.get_training_inputs_auto(set_name)
    transactions = [*transactions]
    
    return transactions, meta_name

def query_meta_all(uri):
    test = query_meta(uri, 'test')
    meta_pk = test[1]
    test=test[0]
    validation = query_meta(uri, 'validation')[0]
    test+= validation
    test = [ np.array(t) for t in test]
    test = np.array(test)

    return test, meta_pk

def separate(datas, output_engine_pk):
    raws = datas[:,range(len(datas[0])-1)]
    engines_results = datas[:,-1]

    inputs = [[] for _ in range(len(engines_results))]
    predictions=[]

    for idx,r in enumerate(engines_results):
        for e in r:
            if e[0] != output_engine_pk:
                inputs[idx].append(e)
            else:
                predictions.append(e)

    return raws, inputs, predictions

def preprocess_meta(raws, inputs, name_file="xy.npz", name_originals='originals.npz', save=False):
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
        ys.append(raws[idx][15])

    if save:
        np.savez(name_file, x=xs, y_account_decoded=ys, account_decoded=encoder)
        np.savez(name_originals, originals=raws)
    
    return xs, ys, encoder

    
