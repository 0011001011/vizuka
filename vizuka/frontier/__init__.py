import os
import pkgutil
import inspect
import importlib
import pickle

from vizuka.frontier import (
        similarity,
        deterministic,
        bhattacharyya,
        )

from vizuka.plugins import frontier as frontier_plugins


def list_frontiers():
    built_in_similarity= {
            'bhattacharyya' :bhattacharyya.Bhattacharyya,
            'all'           :deterministic.AllFrontiers,
            'none'          :deterministic.NoneFrontiers,
            }

    extra_similarity = {}
    
    for (module_loader, name, ispkg) in pkgutil.iter_modules(frontier_plugins.__path__):
        plugin = importlib.import_module('.plugins.frontier.'+name, 'vizuka')
        members = inspect.getmembers(plugin)
        for _, class_ in members:
            if inspect.isclass(class_):
                if issubclass(class_, similarity.Similarity):
                    extra_similarity[name] = class_

    return built_in_similarity, extra_similarity
    

def make_frontier(method):
    built_in_similarity, extra_similarity = list_frontiers()
    available_similarity = {**built_in_similarity, **extra_similarity}
    
    similarity_builder = available_similarity.get(method, None)
    
    return similarity_builder()
