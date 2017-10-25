import os
import pkgutil
import inspect
import importlib
import pickle
import logging

from vizuka.clustering import (
        kMeans,
        dbscan,
        dummy,
        clusterizer,
        )

from vizuka.plugins import clustering as clustering_plugins

def get_required_arguments(method_name):
    if method_name=='dummy':
        return []
    else:
        builtin, extra = list_clusterizer()
        available_clusterizers = {**builtin, **extra}
        return inspect.getargspec(available_clusterizers[method_name]).args[1:]

def list_clusterizer():
    built_in_clusterizers = {
            'kmeans':kMeans.KmeansClusterizer,
            'dbscan':dbscan.DBSCANClusterizer,
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
    

def make_clusterizer(method='dummy', **kwargs):
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
    
    required_parameters = inspect.getargspec(clusterizer_builder).args
    given_parameters    = {name:value for name,value in kwargs.items() if name in required_parameters}
    logging.info("clusterizer: building a clustering engine with parameters:\n{}".format(given_parameters))
    clusterizer = clusterizer_builder(**given_parameters)
    
    return clusterizer
