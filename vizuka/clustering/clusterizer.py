'''
Clustering engine to use with Vizualization

If you want to implement one, dont forget to add it
on qt_handler, to be able to select on the IHM
Also, adds it in make_clusterizer obviously

3 methods are necessary to implement, cf Clusterizer():
    init    - set main params
    fit     - to prepare the algo for the data
    predict - to find out the cluster of a (x,y)
'''
import pickle
import os
import logging

from matplotlib import pyplot as plt

from vizuka.config import CACHE_PATH

NAME_VALUE_SEPARATOR = '::'
PARAMETERS_SEPARATOR = '#'

class Clusterizer():

        def get_param_from_name(name):
            """
            Returns the method and parameters of the clusterizer
            from its name
            """
            params = {}
            blocks = name.split(PARAMETERS_SEPARATOR)
            method = blocks[0]
            for block in blocks[1:]:
                if block:
                    print(block)
                    name, value  = block.split(NAME_VALUE_SEPARATOR)
                    params[name] = value
            return method, params

        def __init__(self, **kwargs):
            """
            Builds a clusterizer object e.g: kmeans
            Do not know the datas at this point, just pass it the bare
            minimum to initialize an engine.

            :param *args **kwargs: parameters passed to the clusterizer engine
                                   it can literally be whatever you want
            """
            pass

        def register_parameters(self, parameters):
            """
            Register the parameters of the clusterizer to generate a name
            The name will be used for saving clusterizer in cache file (pickle)
            and replay session with Load clusterization in Vizuka main window

            :param parameters: a dict with the parameters of the clustering engine
            """
            self.parameters = parameters

        def get_name(self):
            """
            Returns a unique name for the clustering engine
            This name alone is not sufficient to find which cached clustering engine to
            reload as different datasets have a different clustering
            """
            ordered_parameters_name = list(self.parameters)
            ordered_parameters_name.sort()

            return ''.join([
                    self.method,PARAMETERS_SEPARATOR,
                    *[ str(name)+NAME_VALUE_SEPARATOR+str(self.parameters[name])+PARAMETERS_SEPARATOR for name in ordered_parameters_name ]
                    ])[:-1]

        def fit(self, xs):
            """
            First time the engine sees the data.
            Depending on the algorithm you may do your magic your own way
            and maybe store new variable in self, maybe store all the
            predicts for each x directly in a dict.

            :param xs: a list containing data to clusterize
            """
            pass


        def predict(self, xs):
            """
            Finds the cluster(s) in which the data is.

            :params xs: array-like of (x,y) points
            :return: array-like of cluster id
            """
            return (0, ) * len(xs)
        
        def loadable(self, version, path=CACHE_PATH):
            filename = ''.join(
                    [self.get_name(), PARAMETERS_SEPARATOR, version,'.pkl'])
            full_path = os.path.join(path, filename)
            
            if os.path.exists(full_path):
                logging.info('cluster: found clusters in {}'.format(full_path))
                return True
            else:
                return False

        def save_cluster(self, version, path=CACHE_PATH):
            filename = ''.join(
                    [self.get_name(), PARAMETERS_SEPARATOR, version,'.pkl'])
            full_path = os.path.join(path, filename)
            logging.info('cluster: saving clusters in {}'.format(full_path))
            with open(full_path, 'wb') as f:
                pickle.dump(self, f)
        

        def load_cluster(self, version, path=CACHE_PATH):
            filename = ''.join(
                    [self.get_name(), PARAMETERS_SEPARATOR, version,'.pkl'])
            full_path = os.path.join(path, filename)
            logging.info('cluster: loading clusters in {}'.format(full_path))
            with open(full_path, 'rb') as f:
                loaded = pickle.load(f)
                self = loaded
                return loaded





if __name__ == '__main__':
    """
    Yes, I like to test my code with __main__
    """
    from vizuka import dim_reduction as dr
    from vizuka import data_loader as dl
    datas_sets, models = dl.load_tSNE()
    datas = datas_sets[50, 1000, 'pca', 15000]

    clusterizer = make_clusterizer(datas, method='kmeans', n_clusters=80)
    f, ax = plot_clusters(datas, clusterizer)

    plt.show()
