
import matplotlib.pyplot as plt
import numpy as np


def add_subplot(fig = None, layout = 'grid'):
    """
    Add a subplot, and adjust the positions of the other subplots appropriately.
    Lifted from this answer: http://stackoverflow.com/a/29962074/851699

    :param fig: The figure, or None to select current figure
    :param layout: 'h' for horizontal layout, 'v' for vertical layout, 'g' for approximately-square grid
    :return: A new axes object
    """
    if fig is None:
        fig = plt.gcf()
    n = len(fig.axes)
    n_rows, n_cols = (1, n+1) if layout in ('h', 'horizontal') else (n+1, 1) if layout in ('v', 'vertical') else \
        vector_length_to_tile_dims(n+1) if layout in ('g', 'grid') else bad_value(layout)
    for i in range(n):
        fig.axes[i].change_geometry(n_rows, n_cols, i+1)
    ax = fig.add_subplot(n_rows, n_cols, n+1)
    return ax


_subplots = {}


def set_named_subplot(name, fig=None, layout='grid'):
    """
    Set the current axes.  If "name" has been defined, just return that axes, otherwise make a new one.

    :param name: The name of the subplot
    :param fig: The figure, or None to select current figure
    :param layout: 'h' for horizontal layout, 'v' for vertical layout, 'g' for approximately-square grid
    :return: An axes object
    """
    if name in _subplots:
        plt.subplot(_subplots[name])
    else:
        _subplots[name] = add_subplot(fig=fig, layout=layout)
    return _subplots[name]


def vector_length_to_tile_dims(vector_length):
    """
    You have vector_length tiles to put in a 2-D grid.  Find the size
    of the grid that best matches the desired aspect ratio.

    TODO: Actually do this with aspect ratio

    :param vector_length:
    :param desired_aspect_ratio:
    :return: n_rows, n_cols
    """
    n_cols = np.ceil(np.sqrt(vector_length))
    n_rows = np.ceil(vector_length/n_cols)
    grid_shape = int(n_rows), int(n_cols)
    return grid_shape


def bad_value(value, explanation = None):
    """
    :param value: Raise ValueError.  Useful when doing conditional assignment.
    e.g.
    dutch_hand = 'links' if eng_hand=='left' else 'rechts' if eng_hand=='right' else bad_value(eng_hand)
    """
    raise ValueError('Bad Value: %s%s' % (value, ': '+explanation if explanation is not None else ''))
