import os
import pkgutil
import inspect
import importlib
import pickle

from vizuka.similarity import similarity
from vizuka.similarity import bhattacharyya
from vizuka.similarity import deterministic

from vizuka.plugins import similarity as similarity_plugins


def list_similarity():
    built_in_similarity= {
            'bhattacharyya' :bhattacharyya.BhattacharyyaSimilarity,
            'all'           :deterministic.AllFrontiers,
            'none'          :deterministic.NoneFrontiers,
            }

    extra_similarity = {}
    
    for (module_loader, name, ispkg) in pkgutil.iter_modules(similarity_plugins.__path__):
        plugin = importlib.import_module('.plugins.similarity.'+name, 'vizuka')
        members = inspect.getmembers(plugin)
        for _, class_ in members:
            if inspect.isclass(class_):
                if issubclass(class_, similarity.Similarity):
                    extra_similarity[name] = class_

    return built_in_similarity, extra_similarity
    

def make_frontier(method):
    built_in_similarity, extra_similarity = list_similarity()
    available_similarity = {**built_in_similarity, **extra_similarity}
    
    similarity_builder = available_similarity.get(method, None)
    
    return similarity_builder()
