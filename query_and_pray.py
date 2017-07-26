"""
Query some datas directly from databases
Then prepare it for the vizualization
It should be launced only once in a while
"""
import numpy as n

uri = 'postgres://ds:ds@localhost/gold_standard_manakin'

def query_meta(uri, set_name):
    import meta
    MDI = meta.algorithm_database_interface.MetaDatabaseInterface(uri)
    transactions = MDI.get_training_inputs_auto(set_name)
    np.array(list(transactions))

    return transactions

def query_meta_all(uri):
    test = query_meta(uri, 'test')
    validation = query_meta(uri, 'validation')
    test.append(validation)
    test = np.array(test)

    return test

def separate(datas):
    raws = datas[:,range(datas.shape[1]-1)]
    inputs = datas[:,-1]

    return raws, inputs

def preprocess_meta(raws, inputs, save=False):
    class_predicted = set()
    class_existing  = set()
    engines = set()

    for input_ in inputs:
        for suggestion in input_:
            class_predicted.add(suggestion[2])
            engines.add(suggestion[0])
    engines = list(engines)

    for transaction in raws:
        class_existing.add(transaction[-5])

    all_class = class_predicted.union(class_existing)

    # builds encoder for the y
    encoder ={}
    encoding_compress = []
    blank_y = np.array([False]*len(all_class), dtype=bool)
    for idx, class_ in all_class:
        encoding = blank_y.copy()
        encoding[idx] = True
        encoder[class_] = encoding
        encoding_compress.append(class_)

    # preprocess the x
    blank_x = np.array([0]*len(blank_y)*len(engines), dtype=float)
    xs=[]
    ys=[]
    for transaction in inputs:
        x = blank_x.copy()
        for prediction in transaction[-1]:
            label = prediction[0]
            idx = engines.index(label)*len(blank_y)+encoding_compress.index(label)
            x[idx] = prediction[3]
        xs.append(x)
        ys.append(transaction[15])

    if save:
        pass
    
    return xs, ys, encoder

    
