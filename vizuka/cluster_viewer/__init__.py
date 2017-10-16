import importlib
import pkgutil
import inspect

from vizuka.plugins import cluster_viewer as plotter_plugins
from vizuka.cluster_viewer import (
        image,
        density,
        counter,
        wordcloud,
        )


def list_plotter():
    built_in_plotters = {
            'images':image.RandomImages,
            'density':density.Density,
            'logdensity':density.LogDensity,
            'counter':counter.Counter,
            'wordcloud':wordcloud.Wordcloud,
            }

    extra_plotters = {}
    
    for (module_loader, name, ispkg) in pkgutil.iter_modules(plotter_plugins.__path__):
        plugin = importlib.import_module('.plugins.cluster_viewer.'+name, 'vizuka')
        members = inspect.getmembers(plugin)
        for _, class_ in members:
            if class_ in plotter.Plotter.__subclasses__():
                extra_plotters[name] = class_

    return built_in_plotters, extra_plotters
    

def make_plotter(method):
    """
    Returns an axe with the data plotted as you requested

    :param: data is a list of the observations
    :param: fig is the Figure on which the axe will be drawn
    :param:spec is the GridSpec for the axe

    :return: the axe with the visualization
    """
    
    built_in_plotters, extra_plotters = list_plotter()
    available_plotters = {**built_in_plotters, **extra_plotters}
    
    selected_plotter = available_plotters.get(method, None)

    return selected_plotter()
