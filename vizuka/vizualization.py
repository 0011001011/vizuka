"""
Module built around the class Vizualization
It is the main module that draws a nice IHM interface to explore your data

See class Vizualization
"""
import sys
import os
import logging
from collections import Counter
import pickle
import csv

from scipy import stats
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # noqa
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt

from vizuka.helpers import viz_helper
from vizuka.graphics import drawing
from vizuka import clustering
from vizuka import frontier
from vizuka import data_loader
from vizuka import heatmap
from vizuka.graphics.qt_handler import Cluster_viewer
from vizuka.graphics.qt_handler import Viz_handler
from vizuka.config import (
        path_builder,
        BASE_PATH,
        )


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
            heatmaps_requested = ['entropy', 'accuracy'],
            class_decoder=(lambda x: x), class_encoder=(lambda x: x),
            output_path='output.csv',
            base_path=BASE_PATH,
            version='',
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
        
        (
                self.data_path,
                self.reduced_data_path,
                self.model_path,
                self.graph_path,
                self.cache_path, 
                self.saved_clusters_path,
                
                ) = path_builder(base_path)

        self.predictors = [name for name in os.listdir(self.model_path) if version in name]
        self.output_path = output_path
        self.saved_clusters = [f for f in os.listdir(self.saved_clusters_path) if version in f]
        self.version = version
        
        def str_with_default_value(value):
            if not value:
                return str(special_class)
            return str(value)

        self.correct_outputs = [str_with_default_value(correct_output) for correct_output in correct_outputs]
        self.correct_outputs_original = self.correct_outputs
        self.prediction_outputs = [str_with_default_value(predicted_output) for predicted_output in predicted_outputs]
        self.prediction_outputs_original = self.prediction_outputs
        
        self.projected_input = projected_input
        self.projected_input_original = self.projected_input
        self.x_raw = raw_inputs
        self.x_raw_original = self.x_raw
        self.x_raw_columns = raw_inputs_columns
        self.nb_of_clusters = nb_of_clusters
        self.class_decoder = class_decoder
        self.special_class = str(special_class)
        self.heatmaps_requested = heatmaps_requested
        
        self.features_name_to_filter = features_name_to_filter
        self.correct_class_to_display = {}
        self.predicted_class_to_display = {}
        self.feature_to_display_by_col = {}
        self.features_to_display = features_name_to_display
        self.cluster_view_selected_indexes = []
        self.filters = {
                'PREDICTIONS':set(),
                'GROUND_TRUTH':set(),
                **{k:set() for k in features_name_to_filter},
                }
        self.left_clicks = set()
        self.selected_clusters = set()

        #self.possible_outputs_list = list({self.class_decoder(y_encoded) for y_encoded in self.correct_outputs})
        self.possible_outputs_set = set(self.correct_outputs).union(set(self.prediction_outputs))
        self.possible_outputs_set.discard(None)
        self.possible_outputs_list = list(self.possible_outputs_set)
        self.possible_outputs_list.sort()
        # logging.info("correct outputs : %", self.correct_outputs)
        self.projection_points_list_by_correct_output = {y: [] for y in self.correct_outputs}
        self.index_by_true_output = {class_:[] for class_ in self.possible_outputs_list}
        self.nb_of_individual_by_true_output = {}

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
        self.local_accuracy = {}
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

        self.amplitude = viz_helper.find_amplitude(self.projected_input_original)

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
        self.clusterizer = clustering.make_clusterizer(
                xs=self.projected_input_original,
                nb_of_clusters=self.nb_of_clusters,
                mesh=self.mesh_centroids,
                method='dummy',
                )
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
        ) = viz_helper.separate_prediction(
                self.prediction_outputs,
                self.correct_outputs,
                self.special_class,
                )

        (
            self.well_predicted_projected_points_array,
            self.misspredicted_projected_points_array,
            self.not_predicted_projected_points_array,

            ) = viz_helper.get_projections_from_index(
                    self.index_good_predicted,
                    self.index_bad_predicted,
                    self.index_not_predicted,
                    self.projected_input,
                    )

        self.accuracy_by_class = viz_helper.get_accuracy_by_class(
                self.index_by_true_output,
                self.correct_outputs,
                self.prediction_outputs,
                self.possible_outputs_list)

        
    def reload_predict(self, filename):
        """
        Call this function if the predictor set has changed
        filename: the name of the predictions file to load, should
        be located in the self.model_path folder
        """

        self.prediction_outputs = data_loader.load_predict_byname(filename, path=self.model_path)
        self.prediction_outputs_originals = self.prediction_outputs
        self.calculate_prediction_projection_arrays()

        self.print_global_summary(self.global_summary_axe)

        self.init_clusters()
        self.update_all_heatmaps()

        self.reset_summary()
        self.reset_viz()

        for click in self.left_clicks:
            self.do_left_click(click)

        self.refresh_graph()
        


    def filter_by_feature(self, feature_col, selected_feature_list):
        """
        Updates the list of index to display, filter with :

        feature_col: the column of the feature in "originals"
        selected_feature_list: list of checked/unchecked feature to display
        """

        featured_data = [
                item for item, selected in selected_feature_list.items() if selected
                ]
        self.feature_to_display_by_col[feature_col] = featured_data
        self.filters[feature_col] = featured_data

        # self.display_by_filter()
        self.conciliate_filters(self.filters)

    def filter_by_correct_class(self, selected_outputs_class_list):
        """
        Filter by class, on the TRUE class of each point
        """
        self.correct_class_to_display = {
                widget.class_ for widget in selected_outputs_class_list if widget.isChecked()
                }
        self.filters["GROUND_TRUTH"] = self.correct_class_to_display
        # self.display_by_filter()
        self.conciliate_filters(self.filters)

    def filter_by_predicted_class(self, selected_outputs_class_list):
        """
        Filter by class, on the PREDICTED class of each point
        """
        self.predicted_class_to_display = {
                widget.class_ for widget in selected_outputs_class_list if widget.isChecked()
                }
        self.filters["PREDICTIONS"] = self.predicted_class_to_display
        # self.display_by_filter()
        self.conciliate_filters(self.filters)

    def load_only_some_indexes(self, indexes):
        self.prediction_outputs = [self.prediction_outputs_original[i]  for i in indexes]
        self.projected_input    = [self.projected_input_original[i]     for i in indexes]
        self.correct_outputs    = [self.correct_outputs_original[i]     for i in indexes]
        self.x_raw              = [self.x_raw_original[i]               for i in indexes]

        self.projection_points_list_by_correct_output = {y: [] for y in self.correct_outputs}
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

        self.calculate_prediction_projection_arrays()
        self.init_clusters()
        
        self.print_global_summary(self.global_summary_axe)
        self.update_all_heatmaps()
        self.update_summary()
        self.print_summary(self.summary_axe)
        self.update_cluster_view()
    
    def filters_active(self):
        """
        True if some filters are active
        """

        additional_filters_active = False
        other_filters = {k:v for k,v in self.filters.items() if k!="PREDICTIONS" and k!="GROUND_TRUTH"}
        for col in other_filters: # other filters are column number identifying features
            additional_filters_active = additional_filters_active or other_filters[col]

        return self.filters["GROUND_TRUTH"] or self.filters["PREDICTIONS"] or additional_filters_active
        

    def conciliate_filters(self, filters):
        saved_zoom = (self.ax.get_xbound(), self.ax.get_ybound())
        to_display = set(range(len(self.projected_input_original)))

        if filters["GROUND_TRUTH"]:
            filtered = filters["GROUND_TRUTH"]
            to_display = to_display.intersection(set([
                    idx for idx, class_ in enumerate(self.correct_outputs_original) if class_ in filtered]))
            
        if filters["PREDICTIONS"]:
            filtered = filters["PREDICTIONS"]
            to_display = to_display.intersection(set([
                    idx for idx, class_ in enumerate(self.prediction_outputs_original) if class_ in filtered]))
        other_filters = {k:v for k,v in filters.items() if k!="PREDICTIONS" and k!="GROUND_TRUTH"}
        for col in other_filters: # other filters are column number identifying features
            if other_filters[col]:
                to_display.intersection(set([
                    idx for idx, x in enumerate(self.x_raw) if x[col] not in other_filters[col]]))

        viz_helper.remove_pathCollection(self.ax)
        
        self.load_only_some_indexes(to_display)

        drawing.draw_scatterplot(
                self.ax,
                self.well_predicted_projected_points_array,
                self.misspredicted_projected_points_array,
                self.not_predicted_projected_points_array
                )
        self.ax.set_xbound(saved_zoom[0])
        self.ax.set_ybound(saved_zoom[1])

        self.refresh_graph()

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

        if left_click:
            self.do_left_click((x,y))

        elif right_click:
            self.do_right_click((x,y))

        self.refresh_graph()
    
    def do_left_click(self, xy):
        x, y = xy
        if (x is None) or (y is None):  # clicks out of the screen
            return
        
        # Register this click for replay
        self.left_clicks.add((x,y))
        
        # find associated cluster and gather data
        clicked_cluster = self.clusterizer.predict([(x,y)])[0]
        self.selected_clusters.add(clicked_cluster)
        self.delimit_cluster(clicked_cluster, color=self.manual_cluster_color)

        self.update_summary()
        self.print_summary(self.summary_axe)
        
        # shows additional info if requested (argument -s)
        self.update_cluster_view()

            # self.cluster_view.update_cluster_view(
            #        clicked_cluster,
            #        self.index_by_cluster_label,
            #        indexs_good = self.index_good_predicted,
            #        indexes_bad = self.index_bad_predicted,
            #        )

    def do_right_click(self, xy):
            logging.info("Cleaning state, unselecting clusters, redrawing")
            print("Cleaning state, unselecting clusters, redrawing")
            # reboot vizualization
            self.left_clicks = set()
            self.selected_clusters = set()
            self.reset_summary()
            if self.cluster_view:
                self.cluster_view.reset()
            self.reset_viz()

    def reset_viz(self):
        """
        Reset (graphically) the vizualization
        ..note:: does not touch the summary array, for this use self.reset_summary()
        """
        logging.info("scatterplot: removing specific objects")

        self.left_clicks = set()
        for ax in self.axes_needing_borders:
            for i in ax.get_children():
                if isinstance(i, matplotlib.collections.PathCollection):
                    i.remove()
                elif isinstance(i, matplotlib.lines.Line2D) or isinstance(i, matplotlib.collections.LineCollection):
                    if i.get_color() == self.manual_cluster_color:
                        i.remove()
        
        logging.info("scatterplot: drawing observations")
        if self.cluster_view:
            self.clear_cluster_view()

        drawing.draw_scatterplot(
                self.ax,
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
        self.local_accuracy = {}
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
        self.selected_cluster = set()

    def init_clusters(self):
        """
        Clusterize the data and retrieve some countings
        """
        (
            self.index_by_cluster_label,
            self.all_cluster_labels,
            self.nb_of_points_by_class_by_cluster,
            self.cluster_by_idx,

            ) = viz_helper.cluster_label_mesh(

                self.clusterizer,
                self.projected_input,
                self.projection_points_list_by_correct_output,
                self.correct_outputs,
                )


        (
            self.nb_good_point_by_cluster,
            self.nb_bad_point_by_cluster,
            self.nb_good_point_by_class_by_cluster,
            self.nb_bad_point_by_class_by_cluster,
            self.nb_null_point_by_cluster,

            ) = viz_helper.get_count_by_cluster(

                self.all_cluster_labels,
                self.index_by_cluster_label,
                self.correct_outputs,
                self.index_good_predicted,
                self.index_bad_predicted,
                self.index_not_predicted,
                )


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

    

    def get_coordinates_from_index(self, index):
        return (self.resolution - index // self.resolution -1, index % self.resolution)
        # check it's the same
        # return (self.resolution - int(((index - index % self.resolution) / self.resolution)) - 1,
        #         index % self.resolution)


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

        similarity = frontier.make_frontier(method)
        self.similarity_measure = similarity.compare

        if method == 'bhattacharyya':
            self.normalize_frontier=True
        elif method =='all':
            self.normalize_frontier=False
        elif method == 'none':
            self.normalize_frontier=False
            self.refresh_graph()
            return
        
        drawing.apply_borders(
                self,
                self.normalize_frontier,
                self.similarity_measure,
                self.axes_needing_borders
                )
        self.refresh_graph()
        logging.info('frontiers : applied '+method)

    def request_clusterizer(self, method, params):
        """
        Creates the clustering engine depending on the method you require
        Actually different engine use different initialisations

        :param params: the parameters as a dict, of the clustering
        :param method: the name of the clustering engine
        """
        logging.info("cluster: requesting a new " + method + " engine")
        self.clusterizer = clustering.make_clusterizer(
                method = method,
                mesh = self.mesh_centroids,
                **params,
                )

        if (not self.filters_active()) and self.clusterizer.loadable(version=self.version):
            logging.info("cluster: found cached clustering, loading..")
            self.clusterizer = self.clusterizer.load_cluster(version=self.version)
            logging.info("cluster: ready")
        else:
            logging.info("cluster: fitting the clusters")
            self.clusterizer.fit(self.projected_input_original)
            if not self.filters_active():
                logging.info("cluster: saving the clusters")
                self.clusterizer.save_cluster(version=self.version)
            logging.info("cluster: ready")
            
    def request_new_clustering(self, method, params):
        """
        Init and fit a new clustering engin, then update the heatmaps

        :param method: clustering engine to use ..seealso:clustering module
        :param params: a dict with the parameters
        """

        method = method.lower()
        self.request_clusterizer(method, params)

        self.init_clusters()
        self.update_all_heatmaps()

        drawing.apply_borders(
                self,
                self.normalize_frontier,
                self.similarity_measure,
                self.axes_needing_borders)
        logging.info('borders: done')


        self.reset_summary()
        self.reset_viz()
        self.refresh_graph()

    def update_all_heatmaps(self):
        """
        Get all heatmaps registered by request_heatmap and draw them from scratch
        """
        for (this_heatmap, axe, title) in self.heatmaps:
            axe.clear()
            axe.axis('off')

            this_heatmap.update_colors(vizualization=self)
            heatmap_color = this_heatmap.get_all_colors()

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

            axe.set_xlim(-self.amplitude / 2, self.amplitude / 2)
            axe.set_ylim(-self.amplitude / 2, self.amplitude / 2)
            axe.axis('off')
            axe.set_title(title)

        self.refresh_graph()

    def update_summary(self):
        """
        Updates the local variables (the variables containing info about currently
        selected clusters).

        Three objects are important inside the object Vizualization and need to be updated :
            - self.local_classes contains possible_outputs_list inside current_cluster
            - self.local_effectif contains the effetif of each label inside current_cluster
            - self.local_accuracy is the ratio of good/total predicted inside cluster, by label
            - self.local_confusion_by_class is the error made by the predictor, indexed by true output
        """
        effectif_by_class = {}
        logging.info("local_variables=updating")
        
        # Get effectif
        for clr in self.selected_clusters:
            for cls, effectif in self.nb_of_points_by_class_by_cluster.get(clr,{}).items():
                effectif_by_class[cls] = effectif_by_class.get(cls,0) + effectif
        effectif_by_class = {cls:e for cls, e in effectif_by_class.items() if e>0}

        # Set 3 first local variables
        self.local_classes  = list(effectif_by_class.keys())
        self.local_sum      = sum (effectif_by_class.values())
        self.local_effectif = effectif_by_class

        # Counts the good/bad predictions in these clusters
        logging.info("local_variables=counting good/bad")
        nb_good_by_class = {cls:sum([self.nb_good_point_by_class_by_cluster.get(clr,{}).get(cls,0) for clr in self.selected_clusters]) for cls in self.local_classes}
        nb_bad_by_class  = {cls:sum([self.nb_bad_point_by_class_by_cluster.get(clr,{}).get(cls, 0) for clr in self.selected_clusters]) for cls in self.local_classes}

        self.local_bad_count_by_class  = nb_bad_by_class
        self.local_good_count_by_class = nb_good_by_class
        
        # Gets accuracy
        logging.info("local_variables=counting accuracy")
        for cls in self.local_classes:
            self.local_accuracy[cls] = (
                nb_good_by_class.get(cls,0) /
                (
                    nb_good_by_class.get(cls,0)
                    + nb_bad_by_class.get(cls,0)
                    ))
        
        # Get confusion matrix
        logging.info("local_variables=calculating confusion")
        confusion = {}
        for clr in self.selected_clusters:
            for idx in self.index_by_cluster_label.get(clr,{}):
                if idx in self.index_bad_predicted:
                    correct            = self.correct_outputs[idx]
                    prediction         = self.prediction_outputs[idx]
                    confusion[correct] = confusion.get(correct, []) + [prediction]
        
        self.local_confusion_by_class_sorted = {cls:[] for cls in confusion}
        for cls, errors in confusion.items():
            self.local_confusion_by_class_sorted[cls] = Counter(errors).most_common(2)
        logging.info("local_variables=ready")

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

        This method gets its data mainly from self.local_accuracy and self.local_effectif,
        these objects are self-updated when needed and contain the data of the user-selected clusters
        
        :param max_row: max nb of row to add in table summary
        :param axe: the matplotlib axe in which the stats will be plotted
        """
        logging.info("local_summary=printing")
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
                        self.local_accuracy[c]*100,
                        (self.local_accuracy[c]-self.accuracy_by_class[c])*100,
                        stats.binom_test(
                            self.local_effectif[c]*self.local_accuracy[c],
                            self.local_effectif[c],
                            self.accuracy_by_class[c],
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
                        '{:.6}'.format(class_mistaken)+ '({0:.1f}%)'.format(
                            error_count/float(self.local_bad_count_by_class[c])*100
                            )
                        for class_mistaken, error_count in self.local_confusion_by_class_sorted.get(c,[]) if error_count != 0
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
        
        values.append([self.local_sum, '-', len(self.projected_input), ' '])
        row_labels.append('all')

        self.rows = row_labels

        cols = [
                '#class_local (#class_local/#class)',
                'accuracy local (delta accuracy) - p-value',
                 '#class (#class/#all_class)',
                 'common mistakes'
                 ]
        
        axe.clear()
        summary = axe.table(
            cellText=values,
            rowLabels=row_labels,
            colLabels=cols,
            loc='center',
        )
        
        summary.auto_set_font_size(False)
        summary.set_fontsize(8)
        axe.axis('off')

        logging.info("local_summary=ready")
        
    def print_global_summary(self, ax, max_row=9):
        """
        Prints a global summary with the most frequent class and
        their classification accuracy
        """
        logging.info("global_summary=calculating")
        
        ax.clear()
        ax.axis('off')
        cols = ['accuracy', 'effectif']
        most_common_classes = Counter(
                {
                    c:len(self.index_by_true_output.get(c,[])) for c in self.possible_outputs_list
                    }
                ).most_common(max_row)

        row_labels = np.array(most_common_classes)[:,0]
        values = []
        for c in row_labels:
            accuracy = self.accuracy_by_class.get(c,0)*100
            effectif = self.nb_of_individual_by_true_output.get(c,0)
            if len(self.projected_input) == 0:
                proportion_effectif = 0
            else:
                proportion_effectif = effectif / float(len(self.projected_input)) * 100

            values.append(['{0:.2f}'.format(accuracy)+"%", '{0:.0f} ({1:.2f}%)'.format(effectif, proportion_effectif)])

        summary = ax.table(
            cellText=values,
            rowLabels=[' '+str(r[:6])+' ' for r in row_labels],
            colLabels=cols,
            loc='center',
        )
        summary.auto_set_font_size(False)
        summary.set_fontsize(8)
        logging.info("global_summary=ready")
    
    def request_heatmap(self, method, axe, title):
        """
        Draw a heatmap based on a heatmap_builder on an axe

        :param heatmap_builder: a Vizualization parameterless method which returns patches
        :param axe: matplotlib axe object in which the heatmap will be plotted
        """
        heatmap_constructor = heatmap.make_heatmap(
                method=method,
                )
        this_heatmap = heatmap_constructor(
                vizualization=self,
                )

        self.heatmaps.append((this_heatmap, axe, title))

    def export(self, output_path, format='csv'):
        """
        Export your selected data in a .csv file for analysis
        """
        logging.info('exporting:...')
        if output_path == "":
            logging.warn("No export name specified, aborting")
            return

        if self.x_raw:
            columns = [
                *self.x_raw_columns,
                'predicted class',
                'projected coordinates',
                ]
            rows =  [
                        [
                            *self.x_raw[idx],
                            self.prediction_outputs[idx],
                            self.projected_input[idx],
                            ]
                        for idx,c in enumerate(self.cluster_by_idx)
                        if c in self.selected_clusters
                        ]

            to_export =  [columns, *rows]

            if format=='csv':
                wr = csv.writer(open(output_path, 'w'))
                wr.writerows(to_export)

            logging.info('exporting: done')
        else:
            logging.info("nothing to export, no raw data provided!")
    

    def make_clusterization_name(self, name_clusters):
        """
        Filename containing the saved session
        """
        return name_clusters + '#' + self.version + '.pkl'

    def save_clusterization(self, name_clusters='clusters'):
        """
        Basically saves a clusterizer and replays a sequence of leftclicks
        """
        if (' ' in name_clusters) or ('.' in name_clusters):
            self.viz_handler.pop_dialog("Please do not use space or '.' in name!")
            return

        filename  = self.make_clusterization_name(name_clusters)
        session   = (self.version, self.clusterizer, self.left_clicks) 
        full_path = os.path.join(self.saved_clusters_path, filename)

        logging.info("user clusters: saving in {}".format(full_path))
        pickle.dump(
                session,
                open(full_path, 'wb')
                )
        self.viz_handler.user_cluster_menulist.add_items([filename])
        return filename

    def load_clusterization(self, name_clusters):
        """
        Basically loads clustering engine and sequence of left_clicks
        """
        full_path = os.path.join(self.saved_clusters_path, name_clusters)

        version_to_load, clusterizer, left_clicks_to_reproduce = pickle.load(
                open(full_path, 'rb'),
                )
        
        if version_to_load != self.version:
            self.viz_handler.pop_dialog(
                    "Impossible to load clusters : bad VERSION (dataset do not match)")
            return
        
        self.reset_viz()
        self.left_clicks = set()
        self.clusterizer = clusterizer
        self.init_clusters()
        self.reset_summary()

        for left_click in left_clicks_to_reproduce:
            self.do_left_click(left_click)

        self.update_summary()
        self.print_summary(self.summary_axe)
        self.print_global_summary(self.global_summary_axe)
        
        self.refresh_graph()

    
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

    def clear_cluster_view(self):
        for clr_view in self.cluster_view:
            clr_view.clear()

    def add_cluster_view(self, feature_to_display, plotter_name):
        logging.info(
                "cluster_view=adding a feature to display, {}:{}".format(
                    feature_to_display, plotter_name)
                )
        self.cluster_view.append(
                Cluster_viewer(
                    {feature_to_display:plotter_name},
                    self.x_raw,
                    self.x_raw_columns,
                    show_dichotomy=True,
                    )
                )
        self.additional_figures.append(self.cluster_view[-1])
        self.viz_handler.add_figure(
                self.additional_figures[-1],
                "Cluster view: {}".format(plotter_name)
                )
        logging.info("cluster_view=ready")

    def update_cluster_view(self):
        import ipdb
        ipdb.set_trace()
        if self.cluster_view:
            logging.info('cluster_viewer=sorting indexes')
            index_selected  = [
                    idx for clr in self.selected_clusters for idx in self.index_by_cluster_label.get(clr,[])]
            index_good = [idx for idx in index_selected if idx in self.index_good_predicted]
            index_bad  = [idx for idx in index_selected if idx in self.index_bad_predicted ]

            for clr_view in self.cluster_view:
                logging.info('cluster_viewer=plotting sorted indexes')
                clr_view.update_cluster_view(
                        index_good    = index_good,
                        index_bad     = index_bad,
                        data_by_index = self.x_raw,
                        )
            logging.info('cluster_viewer=ready')

    def plot(self):
        """
        Plot the Vizualization, define axes, add scatterplot, buttons, etc..
        """
        self.init_clusters()
        self.main_fig = matplotlib.figure.Figure()
        #self.cluster_view = matplotlib.figure.Figure()
        self.cluster_view       = []
        self.additional_figures = []

        gs=GridSpec(3,4)
        #self.view_details = View_details(self.x_raw)
        self.viz_handler = Viz_handler(
                viz_engine         = self,
                figure             = self.main_fig,
                onclick            = self.onclick,
                additional_filters = self.features_name_to_filter,
                additional_figures = self.additional_figures,
                )

        for feature_to_display,plotter_name in self.features_to_display.items():
            self.add_cluster_view(feature_to_display, plotter_name)

        import ipdb
        ipdb.set_trace()

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
        # contain accuracy of correct prediction and entropy

        self.axes_needing_borders = [self.ax]

        # draw heatmap
        logging.info("heatmap=calculating")
        # self.cluster_view = self.main_fig.add_subplot(gs[1,3])
        

        self.heatmaps = []
        
        self.heatmap0 = self.main_fig.add_subplot(
                gs[1,3],
                sharex=self.ax,
                sharey=self.ax)
        self.request_heatmap(
                method=self.heatmaps_requested[0],
                axe=self.heatmap0,
                title='Heatmap: accuracy correct predictions')
        self.axes_needing_borders.append(self.heatmap0)
        

        self.heatmap1 = self.main_fig.add_subplot(
                gs[0,3],
                sharex=self.ax,
                sharey=self.ax)
        self.request_heatmap(
                method=self.heatmaps_requested[1],
                axe=self.heatmap1,
                title='Heatmap: cross-entropy Cluster-All')
        self.axes_needing_borders.append(self.heatmap1)
        
        self.update_all_heatmaps()
        logging.info("heatmap=ready")

        # draw scatter plot
        self.reset_viz()

        self.request_new_frontiers('none')
        logging.info('Vizualization=ready')
        self.viz_handler.is_ready()

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
        logging.info("refreshing done")
