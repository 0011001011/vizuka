"""
This module provides heatmaps to be drawn on the side of the main map.
Basically it takes a vizualization object and returns an array with 
associated colors.

If you want to provide new ones dont forget to register them in qt_handler
"""

import numpy as np
import logging


def build(vizualization):
    """
    Prepare an 'accuracy' heatmap (good predictions / total effectif)

    This method is a  heatmap_builder returning a list of patches to be plotted somewhere
    Three colors are actually used : red for bad prediction, blue for correct, and green for
        special_class prediction which is a special label defined at the Vizualization.__init__
        (typically the label "0")
    All colors are mixed linearly

    ..seealso:: add_heatmap
    """

    all_colors = [[0 for _ in range(vizualization.resolution)] for _ in range(vizualization.resolution) ]
    centroids_cluster_by_index = vizualization.clusterizer.predict(vizualization.mesh_centroids)
    logging.info('heatmap: drawing accuracy heatmap')

    for index, (x, y) in enumerate(vizualization.mesh_centroids):

        current_centroid_cluster_label = centroids_cluster_by_index[index]

        nb_good_points = vizualization.nb_good_point_by_cluster.get(current_centroid_cluster_label, 0)
        nb_bad_points  = vizualization.nb_bad_point_by_cluster.get (current_centroid_cluster_label, 0)
        nb_null_points = vizualization.nb_null_point_by_cluster.get(current_centroid_cluster_label, 0)

        nb_of_valid_cluster_points = nb_good_points + nb_bad_points

        if nb_of_valid_cluster_points > 0:
            accuracy_correct   = nb_good_points / float(nb_of_valid_cluster_points)
            accuracy_null      = nb_null_points / float(nb_of_valid_cluster_points)
            accuracy_incorrect = 1 - accuracy_correct
        else:
            accuracy_correct = 1
            accuracy_null = 1
            accuracy_incorrect = 1

        red   = accuracy_incorrect
        green = accuracy_null
        blue  = accuracy_correct
        
        x_coordinate, y_coordinate = vizualization.get_coordinates_from_index(index)

        all_colors[x_coordinate][y_coordinate ] = [red, green, blue]

    logging.info('heatmap: accuracy done')
    return all_colors


