import os
import pkgutil
import inspect
import importlib
import pickle

from vizuka.heatmap import heatmap
from vizuka.heatmap import accuracy
from vizuka.heatmap import entropy

from vizuka.plugins import heatmap as heatmap_plugins


def list_heatmap():
    """
    List all the available heatmaps
    First it looks in this directory, then in plugin/heatmaps/
    All heatmaps must inherit vizuka.heatmap.heatmap.Heatmap
    to be detectable
    
    :return: a dict with built-in heatmaps and their respective
             constructor, and a dict with plugins heatmaps and
            their respective constructor
    """
    built_in_heatmap= {
            'accuracy' : accuracy.AccuracyHeatmap,
            'entropy'  : entropy.EntropyHeatmap,
            }

    extra_heatmap = {}
    
    for (module_loader, name, ispkg) in pkgutil.iter_modules(heatmap_plugins.__path__):
        plugin = importlib.import_module('.plugins.heatmap.'+name, 'vizuka')
        members = inspect.getmembers(plugin)
        for _, class_ in members:
            if inspect.isclass(class_):
                if issubclass(class_, heatmap.Heatmap):
                    extra_heatmap[name] = class_

    return built_in_heatmap, extra_heatmap
    

def make_heatmap(method):
    """
    Gets a list of all available heatmap constructors
    and returns the correct one

    :param method: the name of the heatmap you want
    """
    built_in_heatmap, extra_heatmap = list_heatmap()
    available_heatmap = {**built_in_heatmap, **extra_heatmap}
    
    heatmap_builder = available_heatmap.get(method, None)
    
    return heatmap_builder
