import pkgutil
import inspect
import importlib

from vizuka.dimension_reduction import (
        projector,
        pca,
        tsne,
        )

from vizuka.plugins import dimension_reduction as projector_plugins

def list_projectors():
    """
    List all projector available (in vizuka/plugin/dimension_reduction
    and vizuka/dimension_reduction)

    :return: a dict {method_name:projector_class}
    """
    built_in_projectors = {
            'pca':pca.PCA,
            'tsne':tsne.tSNE,
            }

    extra_projectors= {}
    
    for (module_loader, name, ispkg) in pkgutil.iter_modules(projector_plugins.__path__):
        plugin = importlib.import_module('.plugins.dimension_reduction.'+name, 'vizuka')
        members = inspect.getmembers(plugin)
        for _, class_ in members:
            if class_ in projector.Projector.__subclasses__():
                extra_projectors[name] = class_

    return built_in_projectors, extra_projectors
    

def make_projector(method='tsne', **kwargs):
    """
    Return a projector contructor

    :param method: the name of the algo
    :param kwargs: additional argument (e.g: perplexity for tsne)
    """
    
    built_in_projectors, extra_projectors = list_projectors()
    available_projectors= {**built_in_projectors, **extra_projectors}
    
    projector_builder = available_projectors.get(method, None)
    
    return projector_builder
