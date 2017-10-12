"""
Some uninteresting functions you need for the vizualization objects
"""

import math
import logging

import matplotlib
import numpy as np


def remove_pathCollection(ax):
    for child in ax.get_children():
        if isinstance(child, matplotlib.collections.PathCollection):
            child.remove()

def find_grid_position(x, y, resolution, amplitude):
    """
    Gives the int indexes of (x,y) inside the grid matrix
    :param resolution: size of grid (nb of square by row/column)
    :param amplitude:  size of embedded space, as max of axis (rounded as an int)
                       e.g: [[-1,1],[.1,.1] as an amplitude of 2
    """

    z1 = math.floor(x / (amplitude / float(resolution)))
    z2 = math.floor(y / (amplitude / float(resolution)))

    return z1, z2

def find_grid_positions(xys, resolution, amplitude):
        return [
                find_grid_position(
                        xy[0], xy[1],
                        resolution,
                        amplitude,
                        )
                for xy in xys
]

def find_amplitude(projection):
        """
        Find the absolute max of the axis
        """
        x_proj_amplitude = 1 + int(
            max(-min(np.array(projection)[:, 0]), max(np.array(projection)[:, 0])))
        y_proj_amplitude = 1 + int(
            max(-min(np.array(projection)[:, 1]), max(np.array(projection)[:, 1])))
        return 2 * max(x_proj_amplitude, y_proj_amplitude)


def separate_prediction(predicted_outputs, true_outputs, special_class):
    """
    Gives the index of good/bad/not predicted
    :param predicted_outputs: possible_outputs_list predicted, decoded (human-readable)
    :param true_outputs: true possible_outputs_list, decoded  (human-readable)
    :param special_class: label (decoded) of special class (typically 0)

    :return: indexes of (bad predictions, good prediction, special_class predictions)
    :rtype: array(int), array(int), array(int)
    """

    index_bad_predicted = set()
    index_well_predicted = set()
    index_not_predicted = set()
    
    # Sort good / bad / not predictions
    for index, prediction in enumerate(predicted_outputs):
        if special_class == true_outputs[index]:
            index_not_predicted.add(index)
        if prediction == true_outputs[index]:
            index_well_predicted.add(index)
        else:
            index_bad_predicted.add(index)

    return index_bad_predicted, index_well_predicted, index_not_predicted

def get_projections_from_index(index_good, index_bad, index_special, all_projections):
    """
    From indexes, return the corresponding entries in all_projections in
    np.array format
    """

    good_projections = np.array(
        [all_projections[i] for i in index_good])

    bad_projections = np.array(
        [all_projections[i] for i in index_bad])

    special_projections = np.array(
        [all_projections[i] for i in index_special])

    return good_projections, bad_projections, special_projections

def get_accuracy_by_class(
        index_by_ground_truth,
        ground_truth,
        predictions,
        all_classes ):
    """
    Finds the accuracy by class
    """
    
    accuracy_by_class = {}
    for possible_output in all_classes:
        if not possible_output in index_by_ground_truth:
            continue

        list_correct_index = index_by_ground_truth[possible_output]
        if not len(list_correct_index) > 0:
            accuracy_by_class[possible_output] = 0
            continue

        accuracy_by_class[possible_output] = sum(
                [
                    (ground_truth[i] == predictions[i])
                    for i in list_correct_index
                    ]
                ) / float(len(list_correct_index))

    return accuracy_by_class

def cluster_label_mesh(
        clusterizer,
        data,
        data_by_ground_truth,
        ground_truth,
        ):
    """
    Labels the mesh centroids

    Useful for heatmap right after this method. Should be called just
    after a change in clustering method. Parameter is implicitely the
    clusteriser of the vizualization
    """

    cluster_by_idx = clusterizer.predict(data)
    all_cluster_labels = set(cluster_by_idx)
    index_by_cluster_label = { cluster_label:[] for cluster_label in all_cluster_labels }
    
    nb_of_points_by_class_by_cluster = {
            cluster_label: {
                output_class:0 for output_class in data_by_ground_truth }
            for cluster_label in all_cluster_labels
            }
    
    for index, cluster_label in enumerate(cluster_by_idx):
        index_by_cluster_label[cluster_label].append(index)
        nb_of_points_by_class_by_cluster[cluster_label][ground_truth[index]]+=1

    return (
            index_by_cluster_label,
            all_cluster_labels,
            nb_of_points_by_class_by_cluster,
            cluster_by_idx,
            )



def get_count_by_cluster(
        all_cluster_labels,
        index_by_cluster_label,
        ground_truth,
        index_good,
        index_bad,
        index_special):
    """
    Computes various variables based on cluster newly clusterized such as
    indexes of each data in each cluster etc..
    """

    nb_null_point_by_cluster = dict()
    nb_good_point_by_cluster = dict()
    nb_bad_point_by_cluster  = dict()
    nb_good_point_by_class_by_cluster = dict()
    nb_bad_point_by_class_by_cluster  = dict()
    
    logging.info('clustering: analyze each one')
    for cluster_label in all_cluster_labels:
        nb_good_point_by_cluster[cluster_label] = 0
        nb_bad_point_by_cluster[cluster_label]  = 0
        nb_null_point_by_cluster[cluster_label] = 0
        nb_good_point_by_class_by_cluster[cluster_label] = {}
        nb_bad_point_by_class_by_cluster[cluster_label] = {}

        for point_in_cluster_index in index_by_cluster_label[cluster_label]:
            point_correct_output = ground_truth[point_in_cluster_index]
            if point_in_cluster_index in index_good:
                nb_good_point_by_cluster[cluster_label] += 1
                if point_correct_output in nb_good_point_by_class_by_cluster[cluster_label]:
                    nb_good_point_by_class_by_cluster[cluster_label][point_correct_output] += 1
                else:
                    nb_good_point_by_class_by_cluster[cluster_label][point_correct_output] = 1

            elif point_in_cluster_index in index_bad:
                nb_bad_point_by_cluster[cluster_label] += 1
                if point_correct_output in nb_bad_point_by_class_by_cluster[cluster_label]:
                    nb_bad_point_by_class_by_cluster[cluster_label][point_correct_output] += 1
                else:
                    nb_bad_point_by_class_by_cluster[cluster_label][point_correct_output] = 1
            elif point_in_cluster_index in index_special:
                nb_null_point_by_cluster[cluster_label] += 1
            else:
                logging.error("index not in any indexes : %s", point_in_cluster_index)

    
    return (
            nb_good_point_by_cluster,
            nb_bad_point_by_cluster,
            nb_good_point_by_class_by_cluster,
            nb_bad_point_by_class_by_cluster,
            nb_null_point_by_cluster,
            )


