"""
Module built around the class Vizualization
It is the main module that draws a nice IHM interface to explore your data

See class Vizualization
"""
import sys
import os
import time
import logging
import math
import itertools
from collections import Counter

from scipy import stats
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # noqa
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
import pandas as pd
import wordcloud

from vizuka import similarity
from vizuka.qt_handler import Viz_handler
from vizuka.ml_helpers import (
        cross_entropy,
        )
from vizuka import clustering
from vizuka.cluster_diving import Cluster_viewer
from vizuka.config import (
    MODEL_PATH,
    )

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


class Vizualization:

    """
    This class contains all the tools for the vizualiation

    It is created with only the vector of predictions (containing possible_outputs_list) with its decoder function
    Resolution is the size of the grid of the viz, used for heatmap and clustering
    Mouse control and keyboard shortcuts are used (later: QtButtons) ..seealso:: self.controls
    """
    
    def __init__(
            self,
            raw_inputs,
            raw_inputs_columns,
            projected_input,
            predicted_outputs,
            correct_outputs,
            resolution=100,
            special_class='0',
            nb_of_clusters=120,
            features_name_to_filter=[],
            features_name_to_display={},
            class_decoder=(lambda x: x), class_encoder=(lambda x: x),
            output_path='output.csv',
            model_path=MODEL_PATH,
            data_unique_id_string=''
            ):
        """
        Central function, draw heatmap + scatter plot + zoom + annotations on tSNE data

        :param predicted_outputs: vector of predicted possible_outputs_list
        :param correct_outputs: vector of true possible_outputs_list
        :param projected_input: vector of t-SNE projected data (i.e project_input[index] = (x, y)
        :param class_dn_jobs=4, ecoder: function to decode possible_outputs_list machinetohuman
        :param class_encoder: dict to encode possible_outputs_list humantomachine
        """

        logging.info("Vizualization=generating")
        self.last_time = {}

        self.manual_cluster_color = 'cyan'
        self.output_path = output_path
        self.predictors = os.listdir(model_path)
        self.data_unique_id_string = data_unique_id_string
        self.model_path = model_path
        
        def str_with_default_value(value):
            if not value:
                return str(special_class)
            return str(value)

        self.correct_outputs = [str_with_default_value(correct_output) for correct_output in correct_outputs]
        self.prediction_outputs = [str_with_default_value(predicted_output) for predicted_output in predicted_outputs]
        
        self.projected_input = projected_input
        self.x_raw = raw_inputs
        self.x_raw_columns = raw_inputs_columns
        self.nb_of_clusters = nb_of_clusters
        self.class_decoder = class_decoder
        self.special_class = str(special_class)
        
        self.features_name_to_filter = features_name_to_filter
        self.correct_class_to_display = {}
        self.predicted_class_to_display = {}
        self.feature_to_display_by_col = {}
        self.features_to_display = features_name_to_display
        self.cluster_view_selected_indexes = []

        self.last_clusterizer_method = None
        
        #self.possible_outputs_list = list({self.class_decoder(y_encoded) for y_encoded in self.correct_outputs})
        self.possible_outputs_set = set(self.correct_outputs).union(set(self.prediction_outputs))
        self.possible_outputs_set.discard(None)
        self.possible_outputs_list = list(self.possible_outputs_set)
        self.possible_outputs_list.sort()
        # logging.info("correct outputs : %", self.correct_outputs)
        self.projection_points_list_by_correct_output = {y: [] for y in self.correct_outputs}
        self.nb_of_individual_by_true_output = {}
        self.index_by_true_output = {class_:[] for class_ in self.possible_outputs_list}

        for index, projected_input in enumerate(self.projected_input):
            self.projection_points_list_by_correct_output[self.correct_outputs[index]].append(projected_input)
            self.index_by_true_output[self.correct_outputs[index]].append(index)

        # convert dict values to np.array
        for possible_output in self.projection_points_list_by_correct_output:
            self.projection_points_list_by_correct_output[possible_output] = np.array(
                self.projection_points_list_by_correct_output[possible_output])
            self.nb_of_individual_by_true_output[possible_output] = len(
                self.projection_points_list_by_correct_output[possible_output])

        self.resolution = resolution # counts of tiles per row/column

        self.shift_held = False
        self.ctrl_held = False

        self.local_effectif = {}
        self.local_proportion = {}
        self.local_confusion_by_class = {
                class_:
                {
                    class__:0 for class__ in self.possible_outputs_list} for class_ in self.possible_outputs_list
                }
        self.local_bad_count_by_class = {class_:0 for class_ in self.possible_outputs_list}
        self.local_classes = set()
        self.local_sum = 0
        self.currently_selected_cluster = []
        self.cursor_ids = [0]

        # Get the real possible_outputs_list found in true y

        self.amplitude = find_amplitude(self.projected_input)

        mesh = np.meshgrid(
                np.arange(
                    -self.amplitude,
                    self.amplitude,
                    self.amplitude/(self.resolution/2)
                    ),
                np.arange(
                    -self.amplitude,
                    self.amplitude,
                    self.amplitude/(self.resolution/2)
                    ),
                )
        
        self.size_centroid  = 2 * self.amplitude / self.resolution
        self.mesh_centroids = np.c_[mesh[0].ravel(), mesh[1].ravel()]

        self.calculate_prediction_projection_arrays()

        logging.info('clustering engine=fitting')
        self.clusterizer = clustering.DummyClusterizer(mesh=self.mesh_centroids)
        self.clusterizer.fit(self.projected_input)
        logging.info('clustering engine=ready')
        self.normalize_frontier = True
        
    def calculate_prediction_projection_arrays(self):
        """
        Update all the sets which use true/false flags from predictions
        When to use ? -> if you change (or first load) your predictor
        """

        (
            self.index_bad_predicted,
            self.index_good_predicted,
            self.index_not_predicted,
        ) = separate_prediction(
            self.prediction_outputs,
            self.correct_outputs,
            self.special_class,
        )
    
        # Sort good/bad/not predictions in t-SNE space
        logging.info("projections=listing")
        
        self.proportion_by_class = {}
        for possible_output in self.possible_outputs_list:
            if not possible_output in self.index_by_true_output:
                continue
            list_correct_index = self.index_by_true_output[possible_output]
            if not len(list_correct_index) > 0:
                self.proportion_by_class[possible_output] = 0
                continue
            self.proportion_by_class[possible_output] = sum([
                            (self.correct_outputs[i] == self.prediction_outputs[i])
                            for i in list_correct_index]) / float(len(list_correct_index))

        logging.info("projections=sorting")
        self.well_predicted_projected_points_array = np.array(
            [self.projected_input[i] for i in self.index_good_predicted])

        self.misspredicted_projected_points_array = np.array(
            [self.projected_input[i] for i in self.index_bad_predicted])

        self.not_predicted_projected_points_array = np.array(
            [self.projected_input[i] for i in self.index_not_predicted])

        logging.info("projections=ready")
        
    def reload_predict(self, filename):
        """
        Call this function if the predictor set has changed
        filename: the name of the predictions file to load, should
        be located in the self.model_path folder
        """

        self.prediction_outputs = np.load(os.path.join(self.model_path, filename))['pred']
        self.calculate_prediction_projection_arrays() # us

        self.reset_viz()
        self.refresh_graph()
    
    def draw_scatterplot(self, well_predicted_array, badly_predicted_array, special_class_array):
        """
        Draw the datas on the main 2D map
        """
        if len(well_predicted_array) > 0:
            self.ax.scatter(
                x=well_predicted_array[:, 0],
                y=well_predicted_array[:, 1],
                color='b', marker="+"
            )
        if len(badly_predicted_array) > 0:
            self.ax.scatter(
                x=badly_predicted_array[:, 0],
                y=badly_predicted_array[:, 1],
                color='r',
                marker='+'
            )
        if len(special_class_array) > 0:
            self.ax.scatter(
                x=special_class_array[:, 0],
                y=special_class_array[:, 1],
                marker='x',
                color='g'
            )

    def filter_by_feature(self, feature_col, selected_feature_list):
        """
        Updates the list of index to display, filter with :

        feature_col: the column of the feature in "originals"
        selected_feature_list: list of checked/unchecked feature to display
        """

        self.feature_to_display_by_col[feature_col] = [
                item for item, selected in selected_feature_list.items() if selected
                ]
        self.display_by_filter()

    def filter_by_correct_class(self, selected_outputs_class_list):
        """
        Filter by class, on the TRUE class of each point
        """
        self.correct_class_to_display = {
                output_class for output_class, selected in selected_outputs_class_list.items() if selected
                }
        self.display_by_filter()

    def filter_by_predicted_class(self, selected_outputs_class_list):
        """
        Filter by class, on the PREDICTED class of each point
        """
        self.predicted_class_to_display = {
                output_class for output_class, selected in selected_outputs_class_list.items() if selected
                }
        self.display_by_filter()

    def display_by_filter(self):
        """
        Sum up all the filters and display accordingly
        If none are set to be display in one of the sets, it displays all
        """
        all_unchecked = (
                (not self.predicted_class_to_display) and (not self.correct_class_to_display)
                )
        index_inputs_to_display = set()

        for output_class in self.predicted_class_to_display:
            for index, predicted_class in enumerate(self.prediction_outputs):
                if predicted_class == output_class:
                    index_inputs_to_display.add(index)
        
        for output_class in self.correct_class_to_display:
            for index, true_class in enumerate(self.correct_outputs) :
                if true_class == output_class:
                    index_inputs_to_display.add(index)
    
        for child in self.ax.get_children():
            if isinstance(child, matplotlib.collections.PathCollection):
                child.remove()

        bad_to_display, good_to_display, special_to_display = set(), set(), set()

        for index in index_inputs_to_display:
            index_pass_feature_check = True
            for col in self.feature_to_display_by_col.keys():
                # check if there is a feature check
                if self.x_raw[index][col] not in self.feature_to_display_by_col[col]:
                    index_pass_feature_check = False
                    continue
            if index_pass_feature_check:
                if index in self.index_bad_predicted:
                    bad_to_display.add(index)
                elif index in self.index_good_predicted:
                    good_to_display.add(index)
                else:
                    special_to_display.add(index)

        bad_to_display_array = np.array([self.projected_input[i] for i in bad_to_display])
        good_to_display_array = np.array([self.projected_input[i] for i in good_to_display])
        special_to_display_array = np.array([self.projected_input[i] for i in special_to_display])

        self.draw_scatterplot(
                good_to_display_array,
                bad_to_display_array,
                special_to_display_array
                )

        if all_unchecked:
            # display nothing ? gross

            well_predicted = self.filter_indexes_list_by_features(self.index_good_predicted)
            miss_predicted = self.filter_indexes_list_by_features(self.index_bad_predicted)
            not_predicted = self.filter_indexes_list_by_features(self.index_not_predicted)

            self.draw_scatterplot(
                well_predicted,
                miss_predicted,
                not_predicted,
                )

        self.refresh_graph()

    def filter_indexes_list_by_features(self, indexes_set):
        logging.info("Filtering by feature")
        to_display = indexes_set.copy()
        for col in self.feature_to_display_by_col:
            logging.info("filtering by: {}".format(col))
            for index in indexes_set:
                if self.x_raw[index][col] not in self.feature_to_display_by_col[col]:
                    to_display.discard(index)
        points_to_display = np.array([self.projected_input[i] for i in to_display])
        return points_to_display

    def onmodifier_press(self, event):
        if event.key == 'shift':
            self.shift_held = True
            logging.info("shift held")
        if event.key == 'ctrl':
            self.ctrl_held = True
            logging.info("ctrl held")

    def onmodifier_release(self, event):
        if event.key == 'shift':
            self.shift_held = False
            logging.info("shift unheld")
        if event.key == 'ctrl':
            self.ctrl_held = False
            logging.info("ctrl unheld")

    #######################################
    # Similarity functions to draw clusters

    def onclick(self, event):
        """
        Mouse event handler

        Actions on mouse button pressed
            1 : select a tile (and a class)
            2 : find similar tiles
            3 : reset vizualization (graph+summary)
            
        """
        x = event.xdata
        y = event.ydata
        button = event.button
        left_click, right_click = ((button == 1), (button == 3))

        self.summary_axe.clear()
        self.summary_axe.axis('off')
        

        def handle_left_click(self):
            if (x is None) or (y is None):  # clicks out of the screen
                return
            clicked_cluster = self.clusterizer.predict([(x,y)])[0]
    
            self.delimit_cluster(clicked_cluster, color=self.manual_cluster_color)
            self.update_summary(clicked_cluster)
            self.print_summary(self.summary_axe)
            if self.cluster_view:
                self.cluster_view.update_cluster_view(
                        clicked_cluster,
                        self.index_by_cluster_label,
                        indexes_good = self.index_good_predicted,
                        indexes_bad = self.index_bad_predicted,
                        )

        if left_click:
            handle_left_click(self)
            
        elif right_click:
            # reboot vizualization
            self.reset_summary()
            self.reset_viz()

        self.refresh_graph()

    def reset_viz(self):
        """
        Reset (graphically) the vizualization
        ..note:: does not touch the summary array, for this use self.reset_summary()
        """
        logging.info("scatterplot: removing specific objects")
        for ax in self.axes_needing_borders:
            for i in ax.get_children():
                if isinstance(i, matplotlib.collections.PathCollection):
                    i.remove()
                elif isinstance(i, matplotlib.lines.Line2D) or isinstance(i, matplotlib.collections.LineCollection):
                    if i.get_color() == self.manual_cluster_color:
                        i.remove()
        
        logging.info("scatterplot: drawing observations")
        if self.cluster_view:
            self.cluster_view.clear()

        self.draw_scatterplot(
                self.well_predicted_projected_points_array,
                self.misspredicted_projected_points_array,
                self.not_predicted_projected_points_array
                )
        logging.info("scatterplot: ready")

    def reset_summary(self):
        """
        Reset the local summary
        """
        self.local_effectif = {}
        self.local_proportion = {}
        self.local_confusion_by_class = {
                output_class: {
                    other_output_class:0 for other_output_class in self.possible_outputs_list
                    }
                for output_class in self.possible_outputs_list
                }
        self.local_bad_count_by_class = {
                output_class:0 for output_class in self.possible_outputs_list
                }
        self.local_classes = set()
        self.local_sum = 0
        self.currently_selected_cluster = []

    def cluster_label_mesh(self):
        """
        Labels the mesh centroids

        Useful for heatmap right after this method. Should be called just
        after a change in clustering method. Parameter is implicitely the
        clusteriser of the vizualization
        """

        self.cluster_by_idx = self.clusterizer.predict(self.projected_input)

        all_cluster_labels = set(self.cluster_by_idx)
        
        index_by_cluster_label = { cluster_label:[] for cluster_label in all_cluster_labels }
        
        nb_of_points_by_class_by_cluster = {
                cluster_label: {
                    output_class:0 for output_class in self.projection_points_list_by_correct_output }
                for cluster_label in all_cluster_labels
                }
        
        nb_null_point_by_cluster = dict()
        nb_good_point_by_cluster = dict()
        nb_bad_point_by_cluster  = dict()
        nb_good_point_by_class_by_cluster = dict()
        nb_bad_point_by_class_by_cluster  = dict()
        
        for index, cluster_label in enumerate(self.cluster_by_idx):
            index_by_cluster_label[cluster_label].append(index)
            nb_of_points_by_class_by_cluster[cluster_label][self.correct_outputs[index]]+=1


        logging.info('clustering: analyze each one')
        for cluster_label in all_cluster_labels:
            nb_good_point_by_cluster[cluster_label] = 0
            nb_bad_point_by_cluster[cluster_label]  = 0
            nb_null_point_by_cluster[cluster_label] = 0
            nb_good_point_by_class_by_cluster[cluster_label] = {}
            nb_bad_point_by_class_by_cluster[cluster_label] = {}

            for point_in_cluster_index in index_by_cluster_label[cluster_label]:
                point_correct_output = self.correct_outputs[point_in_cluster_index]
                if point_in_cluster_index in self.index_good_predicted:
                    nb_good_point_by_cluster[cluster_label] += 1
                    if point_correct_output in nb_good_point_by_class_by_cluster[cluster_label]:
                        nb_good_point_by_class_by_cluster[cluster_label][point_correct_output] += 1
                    else:
                        nb_good_point_by_class_by_cluster[cluster_label][point_correct_output] = 1

                elif point_in_cluster_index in self.index_bad_predicted:
                    nb_bad_point_by_cluster[cluster_label] += 1
                    if point_correct_output in nb_bad_point_by_class_by_cluster[cluster_label]:
                        nb_bad_point_by_class_by_cluster[cluster_label][point_correct_output] += 1
                    else:
                        nb_bad_point_by_class_by_cluster[cluster_label][point_correct_output] = 1
                elif point_in_cluster_index in self.index_not_predicted:
                    nb_null_point_by_cluster[cluster_label] += 1
                else:
                    logging.error("index not in any indexes : %s", point_in_cluster_index)

        
        self.nb_good_point_by_cluster = nb_good_point_by_cluster
        self.nb_bad_point_by_cluster = nb_bad_point_by_cluster
        self.nb_good_point_by_class_by_cluster = nb_good_point_by_class_by_cluster
        self.nb_bad_point_by_class_by_cluster = nb_bad_point_by_class_by_cluster
        self.nb_null_point_by_cluster = nb_null_point_by_cluster
        self.index_by_cluster_label = index_by_cluster_label
        self.nb_of_points_by_class_by_cluster = nb_of_points_by_class_by_cluster

    def calculate_centroid_coordinates(self, x, y):
        return x + self.resolution * y

    def draw_the_line(self, x_list, y_list, color='b'):
        for axe in self.axes_needing_borders:
            axe.add_artist(matplotlib.lines.Line2D(xdata=x_list, ydata=y_list, color=color))

    def line(self, float_point):
        return (float_point - self.size_centroid / 2, float_point + self.size_centroid / 2)

    def lower_bound(self, float_point, plus_one=None):
        if plus_one:
            return (float_point + self.size_centroid / 2,)
        return (float_point - self.size_centroid / 2,)


    def delimit_cluster(self, cluster, color='b', **kwargs):
        """
        Delimits one cluster by drawing lines around it
        """
        centroids_cluster_by_index = self.clusterizer.predict(self.mesh_centroids)

        cluster_y_list_by_x = [[] for x in range(self.resolution)]
        cluster_x_list_by_y = [[] for y in range(self.resolution)]

        iter_all_coordinates = ((x, y) for x in range(0, self.resolution) for y in range(0, self.resolution))

        for idx, (x, y) in enumerate(iter_all_coordinates):
            if centroids_cluster_by_index[idx] == cluster:
                cluster_y_list_by_x[x].append(y)
                cluster_x_list_by_y[y].append(x)
                
        calculate_coordinates = self.calculate_centroid_coordinates

        def draw_all_lines(self, list_points, swapped_coordinates=False):
            for a in range(0, self.resolution):
                a_line_b_list = list_points[a]
                if not a_line_b_list:
                    continue
                min_b = min(a_line_b_list)
                max_b = max(a_line_b_list)
                no_hole = (max_b - min_b + 1) == len(a_line_b_list)
                if no_hole:
                    if swapped_coordinates:  # swaped if a is y and b is x, not swapt if a is  and b is y
                        b_float_position, a_float_position= self.mesh_centroids[calculate_coordinates(min_b, a)]
                        self.draw_the_line(
                                self.lower_bound(b_float_position),
                                self.line(a_float_position),
                                color=color)
                        b_float_position, a_float_position = self.mesh_centroids[calculate_coordinates(max_b, a)]
                        self.draw_the_line(
                            self.lower_bound(b_float_position, plus_one=True),
                            self.line(a_float_position),
                            color=color)
                    else:
                        a_float_position, b_float_position= self.mesh_centroids[calculate_coordinates(a, min_b)]
                        self.draw_the_line(
                                self.line(a_float_position),
                                self.lower_bound(b_float_position),
                                color=color)
                        a_float_position, b_float_position = self.mesh_centroids[calculate_coordinates(a, max_b)]
                        self.draw_the_line(
                            self.line(a_float_position),
                            self.lower_bound(b_float_position, plus_one=True),
                            color=color)
                else:  # case not convex, which is not often so it's gonna be dirty
                    for b in a_line_b_list:
                        if swapped_coordinates:
                            if (b - 1) not in a_line_b_list:
                                b_float_position, a_float_position = self.mesh_centroids[calculate_coordinates(b, a)]
                                self.draw_the_line(
                                        self.lower_bound(b_float_position),
                                        self.line(a_float_position),
                                        color=color)
                            if (b + 1) not in a_line_b_list:
                                b_float_position, a_float_position = self.mesh_centroids[calculate_coordinates(b, a)]
                                self.draw_the_line(
                                    self.lower_bound(b_float_position, plus_one=True),
                                    self.line(a_float_position),
                                    color=color)
                        else:
                            if (b - 1) not in a_line_b_list:
                                a_float_position, b_float_position = self.mesh_centroids[calculate_coordinates(a, b)]
                                self.draw_the_line(
                                        self.line(a_float_position),
                                        self.lower_bound(b_float_position),
                                        color=color)
                            if (b + 1) not in a_line_b_list:
                                a_float_position, b_float_position = self.mesh_centroids[calculate_coordinates(a, b)]
                                self.draw_the_line(
                                    self.line(a_float_position),
                                    self.lower_bound(b_float_position, plus_one=True),
                                    color=color)
                            
        draw_all_lines(self, cluster_y_list_by_x, swapped_coordinates=True)
        draw_all_lines(self, cluster_x_list_by_y, swapped_coordinates=False)

    
    def apply_borders(self, normalize_frontier, frontier_builder, *args):
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
        centroids_cluster_by_index = self.clusterizer.predict(self.mesh_centroids)
        for index, xy in enumerate(self.mesh_centroids):
            current_centroid_label = centroids_cluster_by_index[index]
            if index > self.resolution:
                label_down_neighbor = centroids_cluster_by_index[index-self.resolution]
                if label_down_neighbor != current_centroid_label:
                    if (label_down_neighbor, current_centroid_label) not in frontier:
                        current_frontier = frontier_builder(
                                    self.nb_of_points_by_class_by_cluster[label_down_neighbor],
                                    self.nb_of_points_by_class_by_cluster[current_centroid_label]
                                    )
                        if current_frontier > -np.inf:
                            frontier[(label_down_neighbor, current_centroid_label)] = current_frontier

            if index % self.resolution > 0:
                label_left_neighbor = centroids_cluster_by_index[index-1]
                if label_left_neighbor != current_centroid_label:
                    if (label_left_neighbor, current_centroid_label) not in frontier:
                        current_frontier = frontier_builder(
                                    self.nb_of_points_by_class_by_cluster[label_left_neighbor],
                                    self.nb_of_points_by_class_by_cluster[current_centroid_label]
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
        for index, (x, y) in enumerate(self.mesh_centroids):

            current_centroid_label = centroids_cluster_by_index[index]

            if index > self.resolution:
                label_down_neighbor = centroids_cluster_by_index[index-self.resolution]
                if label_down_neighbor != current_centroid_label:
                    if (label_down_neighbor, current_centroid_label) in frontier:
                        frontier_density = normalized_frontier[(label_down_neighbor, current_centroid_label)]

                        lines.append(line_dict_maker(
                            xdata = (x-self.size_centroid/2, x+self.size_centroid/2),
                            ydata = (y-self.size_centroid/2, y-self.size_centroid/2),
                            frontier_density=frontier_density))

            if index % self.resolution > 0:
                label_left_neighbor = centroids_cluster_by_index[index-1]
                if label_left_neighbor != current_centroid_label:
                    if (label_left_neighbor, current_centroid_label) in normalized_frontier:
                        frontier_density = normalized_frontier[(label_left_neighbor, current_centroid_label)]

                        lines.append(line_dict_maker(
                            xdata=(x-self.size_centroid/2, x-self.size_centroid/2),
                            ydata=(y-self.size_centroid/2, y+self.size_centroid/2),
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

    def get_coordinates_from_index(self, index):
        return (self.resolution - index // self.resolution -1, index % self.resolution)
        # check it's the same
        # return (self.resolution - int(((index - index % self.resolution) / self.resolution)) - 1,
        #         index % self.resolution)

    def heatmap_proportion(self):
        """
        Prepare the patches for a 'proportion' heatmap (good predictions / total effectif)

        This method is a  heatmap_builder returning a list of patches to be plotted somewhere
        Three colors are actually used : red for bad prediction, blue for correct, and green for
            special_class prediction which is a special label defined at the Vizualization.__init__
            (typically the label "0")
        All colors are mixed linearly

        ..seealso:: add_heatmap
        """

        all_colors = [[0 for _ in range(self.resolution)] for _ in range(self.resolution) ]
        centroids_cluster_by_index = self.clusterizer.predict(self.mesh_centroids)
        logging.info('heatmap: drawing proportion heatmap')

        for index, (x, y) in enumerate(self.mesh_centroids):

            current_centroid_cluster_label = centroids_cluster_by_index[index]

            nb_good_points = self.nb_good_point_by_cluster.get(current_centroid_cluster_label, 0)
            nb_bad_points  = self.nb_bad_point_by_cluster.get (current_centroid_cluster_label, 0)
            nb_null_points = self.nb_null_point_by_cluster.get(current_centroid_cluster_label, 0)

            nb_of_valid_cluster_points = nb_good_points + nb_bad_points

            if nb_of_valid_cluster_points > 0:
                proportion_correct   = nb_good_points / float(nb_of_valid_cluster_points)
                proportion_null      = nb_null_points / float(nb_of_valid_cluster_points)
                proportion_incorrect = 1 - proportion_correct
            else:
                proportion_correct = 1
                proportion_null = 1
                proportion_incorrect = 1

            red   = proportion_incorrect
            green = proportion_null
            blue  = proportion_correct
            
            x_coordinate, y_coordinate = self.get_coordinates_from_index(index)

            all_colors[x_coordinate][y_coordinate ] = [red, green, blue]

        logging.info('heatmap: proportion done')
        return all_colors


    def heatmap_entropy(self):
        """
        Prepares the patches for an entropy heatmap

        This method is a heatmap_builder returning a list of patches to be
        plotted somewhere
        The maximum entropy for the Vizualization is calculated and used as
        normalization parameter,
        The plot is actually a logplot as it is more eye-friendly
        ..seealso:: add_heatmap

        """
        all_colors = [[0 for _ in range(self.resolution)] for _ in range(self.resolution) ]
        logging.info('heatmap entropy: drawing')
        centroids_label = self.clusterizer.predict(self.mesh_centroids)
        
        entropys = []

        ordered_class_list = list(self.nb_of_individual_by_true_output.keys())
        ordered_class_list.sort()
        global_list = []
        for class_ in ordered_class_list:
            global_list.append(self.nb_of_individual_by_true_output.get(class_))
        global_array = np.array(global_list)
        global_entropy = np.log(global_array / np.sum(global_array))

        for index, (x, y) in enumerate(self.mesh_centroids):
    
            current_centroid_label = centroids_label[index]

            nb_of_point_by_class      = self.nb_of_points_by_class_by_cluster.get(current_centroid_label, {})
            nb_of_point_by_class_list = [nb_of_point_by_class.get(class_, 0) for class_ in ordered_class_list]
           
            try:
                cluster_is_empty = not bool(self.index_by_cluster_label[current_centroid_label])
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

        for index, (x, y) in enumerate(self.mesh_centroids):
            if index > len(entropys):
                current_entropy = min_entropys
            else:
                current_entropy = entropys[index]

            normalized_entropy = ((current_entropy - min_entropys) / amplitude_entropys)

            x_index, y_index = self.get_coordinates_from_index(index)
            all_colors[x_index][y_index] = normalized_entropy

        logging.info('heatmap entropy: done')
        return all_colors


    def request_new_frontiers(self, method):
        """
        Init new frontiers based on a similarity mesure between clusters
        """
        method = method.lower()
        logging.info('frontiers : requiring new delimitations '+method)

        for axe in self.axes_needing_borders:
            for child in axe.get_children():
                if isinstance(child, matplotlib.collections.LineCollection):
                    child.remove()
                    logging.info("removing a line collection")

        if method == 'bhattacharyya':
            logging.debug('frontiers: set up to '+method)
            self.similarity_measure = similarity.bhattacharyya
            self.normalize_frontier=True
        elif method =='all':
            self.similarity_measure = similarity.all_are_dissimilar
            self.normalize_frontier=False
            logging.debug('frontiers: set up to '+method)
        elif method == 'none':
            self.similarity_measure = similarity.all_are_similar
            self.normalize_frontier=False
            self.refresh_graph()
            return
        
        self.apply_borders(
                self.normalize_frontier,
                self.similarity_measure,
                self.axes_needing_borders
                )
        self.refresh_graph()
        logging.info('frontiers : applied '+method)

    def get_cache_file_name(self, method):
        """
        Makes a filename for the pickled object of your clusterizer
        """
        base_path, data_unique_id_string, nb_of_clusters, clustering_method = (
            self.model_path,
            self.data_unique_id_string,
            self.nb_of_clusters,
            method,
        )
        if not os.path.exists(base_path):
            # if the base path isn't valid (for whatever reason, saves into dev/null)
            logging.info("path not valid : {}".format(base_path))
            return os.devnull, False
            
        cache_path = os.path.join(base_path, 'cache')
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        cluster_filename = '_cachejoin_'.join(
                [str(x) for x in [
                    data_unique_id_string,
                    nb_of_clusters,
                    clustering_method
                    ]])

        cluster_path = os.path.join(cache_path, cluster_filename)

        if os.path.exists(cluster_path):
            logging.info("loading the clusterizer in {}".format(cluster_path))
            return (cluster_path), True
        
        return cluster_path, False

    def clustering_fit(self, method):
        """
        Creates the clustering engine depending on the method you require
        Actually different engine use different initialisations
        """

        logging.info("cluster: requesting a new " + method + " engine")
        if method is None:
            method = self.last_clusterizer_method
        if method == 'kmeans':
            self.clusterizer = clustering.KmeansClusterizer(
                n_clusters=self.nb_of_clusters,
            )
        elif method == 'dbscan':
            self.clusterizer = clustering.DBSCANClusterizer()
        elif method == 'loader':
            self.clusterizer = clustering.LoaderClusterizer()
        else:
            self.clusterizer = clustering.DummyClusterizer(
                mesh=self.mesh_centroids,
            )

        cache_file_path, loadable = self.get_cache_file_name(method)

        if loadable:
            logging.info("loading clusterizer")
            self.clusterizer.load_cluster(cache_file_path)
        else:
            logging.info("requested clusterizer not found in cache ({}), calculating".format(
                cache_file_path))
            self.last_clusterizer_method = method
            self.clusterizer.fit(xs=self.projected_input)
            self.clusterizer.save_cluster(cache_file_path)

    def request_new_clustering(self, method):
        """
        Init and fit a new clustering engin, then update the heatmaps

        :param method: clustering engine to use ..seealso:clustering module
        """
        method = method.lower()
        self.clustering_fit(method)

        self.cluster_label_mesh()
        self.update_all_heatmaps()

        self.apply_borders(
                self.normalize_frontier,
                self.similarity_measure,
                self.axes_needing_borders)
        logging.info('borders: done')


        self.reset_summary()
        self.reset_viz()
        self.refresh_graph()

    def update_all_heatmaps(self):
        """
        Get all heatmaps registered by add_heatmap and draw them from scratch
        """
        for (heatmap_builder, axe, title) in self.heatmaps:
            axe.clear()
            heatmap_color = heatmap_builder()
            logging.info("heatmaps: drawing in "+str(axe))
            im = axe.imshow(
                    heatmap_color,
                    interpolation='nearest',
                    vmin=0, vmax=1,
                    extent=(
                        -self.amplitude-self.size_centroid/2,
                        self.amplitude-self.size_centroid/2,
                        -self.amplitude-self.size_centroid/2,
                        self.amplitude-self.size_centroid/2
                        ),
                    aspect='auto')

            logging.info("heatmaps: "+str(axe)+" ready")

            axe.set_xlim(-self.amplitude / 2, self.amplitude / 2)
            axe.set_ylim(-self.amplitude / 2, self.amplitude / 2)
            axe.axis('off')
            axe.set_title(title)
            logging.info('heatmap, axe, title {} {} {}'.format(heatmap_builder, axe, title))

        self.refresh_graph()

    def update_summary(self, current_cluster):
        """
        Add the data of cluster (:param x_g:, :param y_g:) to the local-tobeplotted summary

        Three objects are important inside the object Vizualization and need to be updated :
            - self.currently_selected_cluster is the collection of selected tiles
            - self.local_classes contains possible_outputs_list inside current_cluster
            - self.local_effectif contains the effetif of each label inside current_cluster
            - self.local_sum the sum of local_effectif
            - self.local_proportion is the ratio of good/total predicted inside cluster, by label

        :param current_cluster: cluster name selected by click
        """
        to_include = self.nb_of_points_by_class_by_cluster.get(current_cluster, {})
        to_include = { k:to_include[k] for k in to_include if to_include[k]!=0 }

        if current_cluster in self.currently_selected_cluster:
            return
        else:
            self.currently_selected_cluster.append(current_cluster)

        new_rows = set(to_include.keys()) - self.local_classes

        logging.debug("Classes already detected :" + str(self.local_classes))
        logging.debug("Classes detected on new click :" + str(set(to_include.keys())))
        logging.debug("Classes to add to summary :" + str(set(new_rows)))

        rows_to_update = self.local_classes.intersection(set(to_include.keys()))
        self.local_classes = self.local_classes.union(set(to_include.keys()))
        self.local_sum = sum(to_include.values()) + self.local_sum
        
        nb_good_point_by_class = self.nb_good_point_by_class_by_cluster[current_cluster]
        nb_bad_point_by_class = self.nb_bad_point_by_class_by_cluster[current_cluster]

        for output_class in new_rows:
            self.local_effectif[output_class] = to_include[output_class]

            self.local_proportion[output_class] = (
                nb_good_point_by_class.get(output_class,0) /
                (
                    nb_good_point_by_class.get(output_class,0)
                    + nb_bad_point_by_class.get(output_class,0)
                    )
            )
        for cluster in self.currently_selected_cluster:
            for index in self.index_by_cluster_label[cluster]:
                if index in self.index_bad_predicted:
                    current_class = self.correct_outputs[index]
                    self.local_bad_count_by_class[current_class] += 1
                    self.local_confusion_by_class[current_class][self.prediction_outputs[index]]+=1

        self.local_confusion_by_class_sorted = {output_class:[] for output_class in self.local_confusion_by_class}
        for output_class, errors in self.local_confusion_by_class.items():
            self.local_confusion_by_class_sorted[output_class] = Counter(errors).most_common(2)

        for output_class in rows_to_update:
            self.local_proportion[output_class] = ((
                    self.local_proportion[output_class] * self.local_effectif[output_class]
                    + (
                        nb_good_point_by_class.get(output_class,0)
                        / (
                            nb_good_point_by_class.get(output_class,0)
                            + nb_bad_point_by_class.get(output_class,0)
                            )
                        * to_include.get(output_class, 0)
                   )
                ) / (self.local_effectif[output_class] + to_include.get(output_class, 0))
            )

            self.local_effectif[output_class] += (
                    nb_good_point_by_class.get(output_class,0)
                    + nb_bad_point_by_class.get(output_class,0)
                    )

    def get_selected_indexes(self):
        """
        Find indexes of xs in selected clusters
        """
        indexes_selected = []
        for cluster in self.currently_selected_cluster:
            for index in self.index_by_cluster_label[cluster]:
                indexes_selected.append(index)

        return indexes_selected

    def print_summary(self, axe, max_row=15):
        """
        Print a short summary with basic stats (occurrence, classification rate) on an axe

        This method gets its data mainly from self.local_proportion and self.local_effectif,
        these objects are self-updated when needed and contain the data of the user-selected clusters
        
        :param max_row: max nb of row to add in table summary
        :param axe: the matplotlib axe in which the stats will be plotted
        """

        row_labels = list(self.local_classes)

        values = [
            [
                (
                    '{0:.0f} ({1:.1f}%)'.format(
                        self.local_effectif[c],
                        self.local_effectif[c] / self.nb_of_individual_by_true_output[c] * 100)
                    ),
                (
                    '{0:.2f}% ({1:+.1f}%) - {0:.1f}'.format(
                        self.local_proportion[c]*100,
                        (self.local_proportion[c]-self.proportion_by_class[c])*100,
                        stats.binom_test(
                            self.local_effectif[c]*self.local_proportion[c],
                            self.local_effectif[c],
                            self.proportion_by_class[c],
                            alternative='two-sided'
                            ),
                        )
                    ),
                (
                    '{0:.0f} ({1:.1f}%)'.format(
                        self.nb_of_individual_by_true_output[c],
                        self.nb_of_individual_by_true_output[c] / float(len(self.projected_input)) * 100,
                        )
                    ),
                (
                    ' '.join([
                        '{:.6}'.format(class_mistaken)+ '({0:.1f}%)'.format(error_count/float(self.local_bad_count_by_class[c])*100)
                        for class_mistaken, error_count in self.local_confusion_by_class_sorted[c] if error_count != 0
                        #self.local_bad_count_by_class[c]!=0
                        ])
                    ),
            ]
            for c in row_labels
        ]

        arg_sort = np.argsort([self.local_effectif[c] for c in row_labels])

        values = [values[i] for i in arg_sort[::-1]]
        row_labels = [row_labels[i][:6] for i in arg_sort[::-1]]
        # add row "all" for recap :

        max_row    = min(max_row, min(len(values), len(row_labels)))
        values     = values[:max_row]
        row_labels = row_labels[:max_row]
        
        values.append([self.local_sum, .856789, len(self.projected_input), ' '])
        row_labels.append('all')

        self.rows = row_labels

        cols = [
                '#class_local (#class_local/#class)',
                'accuracy local (delta accuracy) - p-value',
                 '#class (#class/#all_class)',
                 'common mistakes'
                 ]

        summary = axe.table(
            cellText=values,
            rowLabels=row_labels,
            colLabels=cols,
            loc='center',
        )
        summary.auto_set_font_size(False)
        summary.set_fontsize(8)
        logging.info("Details=loaded")
        
    def print_global_summary(self, ax, max_row=9):
        """
        Prints a global summary with the most frequent class and
        their classification accuracy
        """
        
        cols = ['accuracy', 'effectif']
        most_common_classes = Counter(
                {
                    c:len(self.index_by_true_output[c]) for c in self.possible_outputs_list
                    }
                ).most_common(max_row)

        row_labels = np.array(most_common_classes)[:,0]
        values = [
            [
                '{0:.2f}'.format(self.proportion_by_class[c]*100)+"%",
                (
                    '{0:.0f} ({1:.2f}%)'.format(
                        self.nb_of_individual_by_true_output[c],
                        self.nb_of_individual_by_true_output[c] / float(len(self.projected_input)) * 100
                        )
                    ),
            ]
            for c in row_labels
        ]

        summary = ax.table(
            cellText=values,
            rowLabels=[r[:6] for r in row_labels],
            colLabels=cols,
            loc='center',
        )
        summary.auto_set_font_size(False)
        summary.set_fontsize(8)
    
    def add_heatmap(self, heatmap_builder, axe, title):
        """
        Draw a heatmap based on a heatmap_builder on an axe

        :param heatmap_builder: a Vizualization parameterless method which returns patches
        :param axe: matplotlib axe object in which the heatmap will be plotted
        """

        self.heatmaps.append((heatmap_builder, axe, title))

    def export(self, output_path, format='csv'):
        """
        Export your selected data in a .csv file for analysis
        """
        logging.info('exporting:...')
        if self.x_raw:
            to_export =  pd.DataFrame(
                        [
                            [*self.x_raw[idx], self.projected_input[idx], self.prediction_outputs[idx]]
                            for idx,c in enumerate(self.cluster_by_idx)
                            if c in self.currently_selected_cluster
                            ],
                        columns = [*self.x_raw_columns, 'projected coordinates', 'predicted class'],
                        )
            if format=='csv':
                to_export.to_csv(output_path)
            if format=='hdf5':
                to_export.to_hdf(output_path, 'data')
            logging.info('exporting: done')
        else:
            logging.info("nothing to export, no raw data provided!")


    
    def view_details_figure(self):
        """
        Deprecated, was used to display a scatterplot of your selected cluster(s)
        Unusable because too laggy and bloated graphics
        """
        logging.info('exporting:...')
        indexes = self.get_selected_indexes()
        self.viz_handler.set_additional_graph(
                self.view_details.update(indexes)
                )
        logging.info('exporting: done')


    def plot(self):
        """
        Plot the Vizualization, define axes, add scatterplot, buttons, etc..
        """
        self.main_fig = matplotlib.figure.Figure()
        #self.cluster_view = matplotlib.figure.Figure()

        gs=GridSpec(3,4)
        if self.features_to_display and self.x_raw and self.x_raw_columns:
            self.cluster_view = Cluster_viewer(
                    self.features_to_display,
                    self.x_raw,
                    self.x_raw_columns,
                    show_dichotomy=True)
            additional_figures = [self.cluster_view]

        else:
            self.cluster_view = None
            additional_figures = []
        
        #self.view_details = View_details(self.x_raw)
        self.viz_handler = Viz_handler(
                viz_engine         = self,
                figure             = self.main_fig,
                onclick            = self.onclick,
                additional_filters = self.features_name_to_filter,
                additional_figures = additional_figures,
                )

        # main subplot with the scatter plot
        self.ax = self.main_fig.add_subplot(gs[:2,:3])
        self.ax_base_title = 'Correct VS incorrect predictions'
        self.ax.set_title(self.ax_base_title)

        # summary_subplot with table of local stats
        self.summary_axe = self.main_fig.add_subplot(gs[2,:3])
        self.summary_axe.axis('off')

        self.global_summary_axe = self.main_fig.add_subplot(gs[2,3])
        self.global_summary_axe.axis('off')
        self.print_global_summary(self.global_summary_axe)

        # heatmap subplots
        # contain proportion of correct prediction and entropy

        self.axes_needing_borders = [self.ax]

        # draw heatmap
        logging.info("heatmap=calculating")
        # self.cluster_view = self.main_fig.add_subplot(gs[1,3])

        self.heatmaps = []
        
        self.heat_proportion = self.main_fig.add_subplot(
                gs[1,3],
                sharex=self.ax,
                sharey=self.ax)
        self.add_heatmap(
                self.heatmap_proportion,
                self.heat_proportion,
                title='Heatmap: proportion correct predictions')
        self.axes_needing_borders.append(self.heat_proportion)
        

        self.heat_entropy = self.main_fig.add_subplot(
                gs[0,3],
                sharex=self.ax,
                sharey=self.ax)
        self.add_heatmap(
                self.heatmap_entropy,
                self.heat_entropy,
                title='Heatmap: cross-entropy Cluster-All')
        self.axes_needing_borders.append(self.heat_entropy)
        
        self.cluster_label_mesh()

        self.update_all_heatmaps()
        logging.info("heatmap=ready")

        # draw scatter plot
        self.reset_viz()

        self.request_new_frontiers('none')
        logging.info('Vizualization=readyy')

    def show(self):
        logging.info('showing main window')
        self.viz_handler.show()
        logging.info('showing cluster diving window')
        #self.cluster_diver.show()
        logging.info("showing done")
        sys.exit(self.viz_handler.app.exec_())

    def refresh_graph(self):
        logging.info('refreshing main window')
        self.viz_handler.refresh()
        logging.info('refreshing cluster diving')
        #self.cluster_diver.refresh()
        logging.info("refreshing done")

    def time_logging(self, message=None):
        """Silly module to do manual profiling"""
        import inspect
        caller_method_name = inspect.stack()[1][0].f_code.co_name

        if not message:
            self.last_time[caller_method_name] = time.time()
        else:
            new_time = time.time()
            previous_time = self.last_time.get(caller_method_name, None)
            if previous_time:
                logging.info("time for %s : %s", message, new_time - previous_time)
            else:
                logging.info("time for %s : error, no previous time registered", message)
            self.last_time[caller_method_name] = new_time
    
