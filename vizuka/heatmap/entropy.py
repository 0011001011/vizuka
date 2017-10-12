import numpy as np
import logging

from vizuka.helpers.ml_helper import cross_entropy


def build(vizualization):
    """
    Prepares an entropy heatmap

    This method is a heatmap_builder returning a list of patches to be
    plotted somewhere
    The maximum entropy for the Vizualization is calculated and used as
    normalization parameter,
    The plot is actually a logplot as it is more eye-friendly
    ..seealso:: add_heatmap

    """
    all_colors = [[0 for _ in range(vizualization.resolution)] for _ in range(vizualization.resolution) ]
    logging.info('heatmap entropy: drawing')
    centroids_label = vizualization.clusterizer.predict(vizualization.mesh_centroids)
    
    entropys = []

    ordered_class_list = list(vizualization.nb_of_individual_by_true_output.keys())
    ordered_class_list.sort()
    global_list = []
    for class_ in ordered_class_list:
        global_list.append(vizualization.nb_of_individual_by_true_output.get(class_))
    global_array = np.array(global_list)
    global_entropy = np.log(global_array / np.sum(global_array))

    for index, (x, y) in enumerate(vizualization.mesh_centroids):

        current_centroid_label = centroids_label[index]

        nb_of_point_by_class      = vizualization.nb_of_points_by_class_by_cluster.get(current_centroid_label, {})
        nb_of_point_by_class_list = [nb_of_point_by_class.get(class_, 0) for class_ in ordered_class_list]
       
        try:
            cluster_is_empty = not bool(vizualization.index_by_cluster_label[current_centroid_label])
        except KeyError:
            cluster_is_empty=True

        if (not nb_of_point_by_class) or cluster_is_empty:
            current_entropy = 0
        else:
            current_entropy = (
                cross_entropy(
                    global_array,
                    nb_of_point_by_class_list,
                    global_entropy=global_entropy
                    )
                )

        entropys.append(current_entropy)

    min_entropys = min(entropys)
    max_entropys = max(entropys)
    amplitude_entropys = max_entropys - min_entropys
    # this exception is poorly handleded right now
    if float(amplitude_entropys)==0.:
        amplitude_entropys = 1
    logging.info(
            'heatmap entropy: max cross-entropy={} min={}'.format(
                max_entropys,
                min_entropys,)
            )

    for index, (x, y) in enumerate(vizualization.mesh_centroids):
        if index > len(entropys):
            current_entropy = min_entropys
        else:
            current_entropy = entropys[index]

        normalized_entropy = ((current_entropy - min_entropys) / amplitude_entropys)

        x_index, y_index = vizualization.get_coordinates_from_index(index)
        all_colors[x_index][y_index] = normalized_entropy

    logging.info('heatmap entropy: done')
    return all_colors

