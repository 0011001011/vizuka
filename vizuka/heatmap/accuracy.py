"""
This module provides heatmaps to be drawn on the side of the main map.
Basically it takes a vizualization object and returns an array with 
associated colors.

If you want to provide new ones dont forget to register them in qt_handler
"""

import logging

from vizuka.heatmap import heatmap

class AccuracyHeatmap(heatmap.Heatmap):

    def __init__(self, vizualization):
        """
        Prepare an 'accuracy' heatmap (good predictions / total effectif). One color by cluster !

        Heatmap showing the accuracy of the prediction, 3 colors are actually used :
            - red for bad prediction
            - blue for correct
            - green for special_class which is a special label defined at the Vizualization.__init__
            (typically the label "0")

        All colors are mixed linearly

        :returns: a dict of colors to be plotted (indexed by x by y)

        """
        self.update_colors(vizualization)

    
    def update_colors(self, vizualization):
        """
        Updates all the colors from a new vizualization object (containing all infos about
        predictions, true class, clusters, etc...)
        """
        self.all_colors = [[0 for _ in range(vizualization.resolution)] for _ in range(vizualization.resolution) ]
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

            self.all_colors[x_coordinate][y_coordinate ] = [red, green, blue]

        logging.info('heatmap: accuracy done')
        
    def get_all_colors(self):
        """
        Returns colors by x by y coordinates
        """
        return self.all_colors


