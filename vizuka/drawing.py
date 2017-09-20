"""
Functions to draw things on some axes
Probably uninteresting
"""

from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import logging


def draw_scatterplot_from_indexes(
        indexes_to_display,
        indexes_bad_predictions,
        indexes_good_predictions,
        indexes_special_predictions,
        data,
        axe):

    good_to_display    = []
    bad_to_display     = []
    special_to_display = []

    for i in indexes_to_display:
        if i in indexes_special_predictions:
            special_to_display.append(data[i])
        elif i in indexes_good_predictions:
            good_to_display.append(data[i])
        elif i in indexes_bad_predictions:
            bad_to_display.append(data[i])

    bad_to_display     = np.array(bad_to_display)
    good_to_display    = np.array(good_to_display)
    special_to_display = np.array(special_to_display)
    
    draw_scatterplot(axe, good_to_display, bad_to_display, special_to_display)



def draw_scatterplot(
        axe,
        well_predicted_array,
        badly_predicted_array,
        special_class_array):
    """
    Draw the datas on the main 2D map
    """
    if len(well_predicted_array) > 0:
        axe.scatter(
            x=well_predicted_array[:, 0],
            y=well_predicted_array[:, 1],
            color='b', marker="+"
        )
    if len(badly_predicted_array) > 0:
        axe.scatter(
            x=badly_predicted_array[:, 0],
            y=badly_predicted_array[:, 1],
            color='r',
            marker='+'
        )
    if len(special_class_array) > 0:
        axe.scatter(
            x=special_class_array[:, 0],
            y=special_class_array[:, 1],
            marker='x',
            color='g'
        )


def apply_borders(vizualization, normalize_frontier, frontier_builder, *args):
    """
    Returns the line to draw the clusters border
    
    :param normalize_frontier: sset to True if the value given by
    the :param frontier_builder: needs some normalization, if True
    it will be set between [0,1]
    :param frontier_builder: function that takes two dicts
    (clusters) and compute a frontier density (typically
    based on a similarity measure)
    :param axes: list of axes to draw the borders
    """
    axes = args[0]
    frontier = {}

    logging.info('borders: calculating')
    centroids_cluster_by_index = vizualization.clusterizer.predict(vizualization.mesh_centroids)
    for index, xy in enumerate(vizualization.mesh_centroids):
        current_centroid_label = centroids_cluster_by_index[index]
        if index > vizualization.resolution:
            label_down_neighbor = centroids_cluster_by_index[index-vizualization.resolution]
            if label_down_neighbor != current_centroid_label:
                if (label_down_neighbor, current_centroid_label) not in frontier:
                    current_frontier = frontier_builder(
                                vizualization.nb_of_points_by_class_by_cluster[label_down_neighbor],
                                vizualization.nb_of_points_by_class_by_cluster[current_centroid_label]
                                )
                    if current_frontier > -np.inf:
                        frontier[(label_down_neighbor, current_centroid_label)] = current_frontier

        if index % vizualization.resolution > 0:
            label_left_neighbor = centroids_cluster_by_index[index-1]
            if label_left_neighbor != current_centroid_label:
                if (label_left_neighbor, current_centroid_label) not in frontier:
                    current_frontier = frontier_builder(
                                vizualization.nb_of_points_by_class_by_cluster[label_left_neighbor],
                                vizualization.nb_of_points_by_class_by_cluster[current_centroid_label]
                                )
                    if current_frontier > -np.inf:
                        frontier[(label_left_neighbor, current_centroid_label)] = current_frontier

    frontier = { key:frontier[key] for key in frontier if frontier[key] != -np.inf }
    
    if normalize_frontier:
        max_frontier = frontier[max(frontier, key=frontier.get)]
        min_frontier = frontier[min(frontier, key=frontier.get)]

        frontier_amplitude = max_frontier - min_frontier

        if frontier_amplitude:
            normalized_frontier = { key:(frontier[key]-min_frontier) / frontier_amplitude for key in frontier }
        else:
            normalized_frontier = frontier
    else:
        normalized_frontier = frontier


    logging.info('borders: cleaning')
    for axe in axes:
        for child in axe.get_children():
            if isinstance(child, matplotlib.collections.LineCollection):
                child.remove()

    def line_dict_maker(xdata, ydata, frontier_density):
        black = (0, 0, 0)
        return {'xdata': xdata,
                'ydata': ydata,
                'color': black,
                'alpha': frontier_density
                }

    lines = []

    logging.info('borders: drawing')
    for index, (x, y) in enumerate(vizualization.mesh_centroids):

        current_centroid_label = centroids_cluster_by_index[index]

        if index > vizualization.resolution:
            label_down_neighbor = centroids_cluster_by_index[index-vizualization.resolution]
            if label_down_neighbor != current_centroid_label:
                if (label_down_neighbor, current_centroid_label) in frontier:
                    frontier_density = normalized_frontier[(label_down_neighbor, current_centroid_label)]

                    lines.append(line_dict_maker(
                        xdata = (x-vizualization.size_centroid/2, x+vizualization.size_centroid/2),
                        ydata = (y-vizualization.size_centroid/2, y-vizualization.size_centroid/2),
                        frontier_density=frontier_density))

        if index % vizualization.resolution > 0:
            label_left_neighbor = centroids_cluster_by_index[index-1]
            if label_left_neighbor != current_centroid_label:
                if (label_left_neighbor, current_centroid_label) in normalized_frontier:
                    frontier_density = normalized_frontier[(label_left_neighbor, current_centroid_label)]

                    lines.append(line_dict_maker(
                        xdata=(x-vizualization.size_centroid/2, x-vizualization.size_centroid/2),
                        ydata=(y-vizualization.size_centroid/2, y+vizualization.size_centroid/2),
                        frontier_density=frontier_density))
    
                    line_collection_lines = [list(zip(elt['xdata'], elt['ydata'])) for elt in lines]
    line_collection_colors = [(*elt['color'], elt['alpha']) for elt in lines]
    
    for axe in axes:
        axe.add_artist(
                matplotlib.collections.LineCollection(
                    line_collection_lines,
                    colors=line_collection_colors
                    )
                )
    logging.info('borders: ready')


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
