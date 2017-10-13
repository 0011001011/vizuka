import os
import pkgutil
import inspect
import importlib
import pickle

from vizuka.clustering import (
        kMeans,
        DBSCAN,
        dummy,
        clusterizer,
        )

from vizuka.plugins import clustering as clustering_plugins

def load_cluster(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def list_clusterizer():
    built_in_clusterizers = {
            'kmeans':kMeans.KmeansClusterizer,
            'dbscan':DBSCAN.DBSCANClusterizer,
            'dummy':dummy.DummyClusterizer,
            }

    extra_cluterizers = {}
    
    for (module_loader, name, ispkg) in pkgutil.iter_modules(clustering_plugins.__path__):
        plugin = importlib.import_module('.plugins.clustering.'+name, 'vizuka')
        members = inspect.getmembers(plugin)
        for _, class_ in members:
            if class_ in clusterizer.Clusterizer.__subclasses__():
                extra_cluterizers[name] = class_

    return built_in_clusterizers, extra_cluterizers
    

def make_clusterizer(xs, method='kmeans', **kwargs):
    """
    Clusterize the data with specified algorithm
    Naively assume you pass the right parameters for the right algo

    :param data:       array with shape (n,2) of inputs to clusterize
    :param method:     algo to use, supported: kmeans, dbscan, dummy
    :param n_clusters: nb of clusters to find (if applicable)

    :return: a clusterizer object (instance of child of Clusterizer())
    """
    
    built_in_clusterizers, extra_clusterizers = list_clusterizer()
    available_clusterizers = {**built_in_clusterizers, **extra_clusterizers}
    
    clusterizer_builder = available_clusterizers.get(method, None)

    if method == 'kmeans':
        clusterizer = clusterizer_builder(kwargs['nb_of_clusters'])
    elif method == 'dbscan':
        clusterizer = clusterizer_builder()
    else:
        clusterizer = clusterizer_builder(
                mesh=kwargs['mesh'],
                )
    
    clusterizer.fit(xs)
    return clusterizer

