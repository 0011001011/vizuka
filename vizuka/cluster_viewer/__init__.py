import importlib
import pkgutil
import inspect

from vizuka.plugins import cluster_viewer as plotter_plugins
from vizuka.cluster_viewer import (
    plotter,
    image,
    density,
    counter,
    #wordcloud,
)


def list_plotter():
    built_in_plotters = {
        'images': image.RandomImages,
        'density': density.Density,
        'logdensity': density.LogDensity,
        'counter': counter.Counter,
        #'wordcloud':wordcloud.Wordcloud,
    }

    extra_plotters = {}

    for (module_loader, name, ispkg) in pkgutil.iter_modules(
            plotter_plugins.__path__):
        plugin = importlib.import_module('.plugins.cluster_viewer.' + name,
                                         'vizuka')
        members = inspect.getmembers(plugin)
        for _, class_ in members:
            if class_ in plotter.Plotter.__subclasses__():
                extra_plotters[name] = class_

    return built_in_plotters, extra_plotters


def make_plotter(method):
    """
    Returns the plotter function, which represents data

    :param method: the name of the plotter (..seealo:: list_plotter)
    ..seealso::  vizuka.cluster_viewer.plotter
    """

    built_in_plotters, extra_plotters = list_plotter()
    available_plotters = {**built_in_plotters, **extra_plotters}

    selected_plotter = available_plotters.get(method, None)

    return selected_plotter()
