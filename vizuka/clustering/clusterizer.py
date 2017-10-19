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

from matplotlib import pyplot as plt
import logging


class Clusterizer():

        required_arguments = []
        # List of parameters needed for the clustering engine (asked in dialog popup)

        def __init__(self, required_arguments={}):
            """
            Builds a clusterizer object e.g: kmeans
            Do not know the datas at this point, just pass it the bare
            minimum to initialize an engine.

            :param *args **kwargs: parameters passed to the clusterizer engine
                                   it can literally be whatever you want
            """
            self.engine = None
            self.method=''
            self.required_arguments = required_arguments

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
        

        def save_cluster(self, path):
            with open(path, 'wb') as f:
                pickle.dump(self, f)




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
