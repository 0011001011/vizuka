"""
Module built around the class Vizualization
"""

import matplotlib
matplotlib.use('Qt5Agg')  # noqa
from matplotlib.gridspec import GridSpec
import os

from data_viz.qt_handler import Viz_handler
from data_viz.ml_helpers import (
        cross_entropy,
        bhattacharyya,
        )
from data_viz import clustering

from data_viz.config import (
    MODEL_PATH,
    )


import logging
import math
import itertools
from collections import Counter

import numpy as np

from matplotlib import pyplot as plt
import pandas as pd


def find_grid_position(x, y, resolution, amplitude):
    """
    Gives the int indexes of (x,y) inside the grid matrix
    :param resolution: size of grid (number of square by row/column)
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


def separate_prediction(y_pred_decoded, y_true_decoded, name_of_void):
    """
    Gives the index of good/bad/not predicted
    :param y_pred_decoded: labels predicted, decoded (human-readable)
    :param y_true_decoded: true labels, decoded  (human-readable)
    :param name_of_void: label (decoded) of special class (typically 0 hence void)

    :return: indexes of (bad predictions, good prediction, name_of_void predictions)
    :rtype: array(int), array(int), array(int)
    """

    index_bad_predicted = set()
    index_good_predicted = set()
    index_not_predicted = set()

    # Sort good / bad / not predictions
    for i, pred in enumerate(y_pred_decoded):
        """
        logging.info(name_of_void)
        logging.info(y_true[i])
        """
        if name_of_void == y_true_decoded[i]:
            index_not_predicted.add(i)
        if pred == y_true_decoded[i]:
            index_good_predicted.add(i)
        else:
            index_bad_predicted.add(i)

    return index_bad_predicted, index_good_predicted, index_not_predicted


class Vizualization:

    """
    This class contains all the tools for the vizualiation

    It is created with only the vector of predictions (containing labels) with its decoder function
    Resolution is the size of the grid of the viz, used for heatmap and clustering
    Mouse control and keyboard shortcuts are used (later: QtButtons) ..seealso:: self.controls
    """

    def __init__(
            self,
            x_raw,
            proj,
            y_pred,
            y_true,
            resolution=100,
            special_class='0',
            n_clusters=120,
            class_decoder=(lambda x: x), class_encoder=(lambda x: x),
            output_path='output.csv',
            model_path=MODEL_PATH
            ):
        """
        Central function, draw heatmap + scatter plot + zoom + annotations on tSNE data

        :param y_pred: vector of predicted labels
        :param y_true: vector of true labels
        :param proj: vector of t-SNE projected data
        :param class_dn_jobs=4, ecoder: function to decode labels machinetohuman
        :param class_encoder: dict to encode labels humantomachine
        """

        logging.info("Vizualization=generating")

        self.manual_cluster_color = 'cyan'
        self.output_path = output_path
        self.predictors = os.listdir(model_path)

        self.y_true_decoded = [str(y) for y in y_true]
        self.y_true_decoded = [y*(y!='None')+'0'*(y=='None') for y in self.y_true_decoded]
        self.y_pred_decoded = [str(y) for y in y_pred]
        self.y_pred_decoded = [y*(y!='None')+'0'*(y=='None') for y in self.y_pred_decoded]
        #self.y_true = [class_encoder[y] for y in self.y_true_decoded]
        #self.y_pred = [class_encoder[y] for y in self.y_pred_decoded]
        self.proj = proj
        self.x_raw = x_raw
        self.n_clusters = n_clusters 
        self.class_decoder = class_decoder
        self.special_class = str(special_class)

        self.last_clusterizer_method = None
        
        #self.labels = list({self.class_decoder(y_encoded) for y_encoded in self.y_true_decoded})
        self.labels = list(set(self.y_true_decoded).union(set(self.y_pred_decoded)))
        try:
            self.labels.remove(None)
        except ValueError:
            pass
        
        self.labels.sort()

        self.proj_by_class = {y: [] for y in self.y_true_decoded}
        self.total_individual = {}
        self.index_by_class = {class_:[] for class_ in self.y_true_decoded}

        for idx, projection in enumerate(self.proj):
            self.proj_by_class[self.y_true_decoded[idx]].append(projection)
            self.index_by_class[self.y_true_decoded[idx]].append(idx)

        # convert dict values to np.array
        for class_ in self.proj_by_class:
            self.proj_by_class[class_] = np.array(self.proj_by_class[class_])
            self.total_individual[class_] = len(self.proj_by_class[class_])

        self.resolution = resolution # counts of tiles per row/column

        self.shift_held = False
        self.ctrl_held = False

        self.cols = ['effectif local', 'accuracy local',
                     'effectif global',
                     'common mistakes']
        self.local_effectif = {}
        self.local_proportion = {}
        self.local_confusion_by_class = { class_:{ class__:0  for class__ in self.labels } for class_ in self.labels }
        self.local_bad_count_by_class = { class_:0 for class_ in self.labels }
        self.local_classes = set()
        self.local_sum = 0
        self.currently_selected_cluster = []
        self.cursor_ids = [0]

        # Get the real labels found in true y

        self.amplitude = find_amplitude(self.proj) 

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


        (
            self.index_bad_predicted,
            self.index_good_predicted,
            self.index_not_predicted,

            ) = separate_prediction(

            self.y_pred_decoded,
            self.y_true_decoded,
            special_class,
        )

        # Sort good/bad/not predictions in t-SNE space
        logging.info("projections=listing")
        self.proportion_by_class = { 
                class_:
                sum([
                    (self.y_true_decoded[i]==self.y_pred_decoded[i])
                    for i in self.index_by_class[class_]
                    ])
                /float(len(self.index_by_class[class_]))
                for class_ in self.labels
                }
        logging.info("projections=sorting")
        self.x_proj_good = np.array([self.proj[i] for i in self.index_good_predicted])
        self.x_proj_bad  = np.array([self.proj[i] for i in self.index_bad_predicted])
        self.x_proj_null = np.array([self.proj[i] for i in self.index_not_predicted])
        logging.info("projections=ready")
        
        #self.clusterizer = clustering.DummyClusterizer(resolution=self.resolution)
        logging.info('clustering engine=fitting')
        self.clusterizer = clustering.DummyClusterizer(resolution=self.resolution)
        self.clusterizer.fit(self.proj)
        logging.info('clustering engine=ready')
        self.normalize_frontier = True
        
    def reload_predict(self, filename, model_path=MODEL_PATH):
        
        y_pred = np.load(
                os.path.join(model_path, filename)
                )['pred']


        self.y_pred_decoded = y_pred
        #self.y_pred = [class_encoder[y] for y in self.y_pred_decoded]

        (
            self.index_bad_predicted,
            self.index_good_predicted,
            self.index_not_predicted,

            ) = separate_prediction(

            self.y_pred_decoded,
            self.y_true_decoded,
            self.special_class,
        )

        self.proportion_by_class = { 
                class_:
                sum([
                    (self.y_true_decoded[i]==self.y_pred_decoded[i])
                    for i in self.index_by_class[class_]
                    ])
                /float(len(self.index_by_class[class_]))
                for class_ in self.labels
                }
        
        logging.info("projections=sorting")
        self.x_proj_good = np.array([self.proj[i] for i in self.index_good_predicted])
        self.x_proj_bad  = np.array([self.proj[i] for i in self.index_bad_predicted])
        self.x_proj_null = np.array([self.proj[i] for i in self.index_not_predicted])
        logging.info("projections=ready")

        self.reset_viz()
        self.refresh_graph()


    #######################################
    # Similarity functions to draw clusters

    def contains_dominant(self, x0y0, xy):
        """
        Checks if two clusters have same dominant label

        :type x0y0: (int, int) cluster coordinates
        :type xy: (int, int) cluster coordinates
        """
        x0, y0 = x0y0
        x, y = xy

        if self.grid_total[x][y] == {} or self.grid_total[x0][y0] == {}:
            return False

        dominant = max(self.grid_total[x][y], key=self.grid_total[x][y].get)
        other_dominant = max(self.grid_total[x0][y0], key=self.grid_total[x0][y0].get)

        return (dominant == other_dominant)

    def comparable_proportion(self, x0y0, xy, diff=0.10):
        """
        Checks if cluster have comparable *proportion*

        :param diff: percent of max difference in proportion for two similar clusters
        :type x0y0: (int, int) cluster coordinates
        :type xy: (int, int) cluster coordinates
        """
        x0, y0 = x0y0
        x, y = xy

        grid_prop_x_y = self.grid_proportion[x][y]
        grid_prop_x0_y0 = self.grid_proportion[x0][y0]

        return (grid_prop_x0_y0 * (1 + diff) < grid_prop_x_y < grid_prop_x0_y0 * (1 - diff))

    def filter_true_class(self, states_by_class):
        all_unchecked = ( 0 == sum(states_by_class.values()) )

        to_scatter = set()
        for class_, state in states_by_class.items():
            if state or all_unchecked:
                to_scatter.add(class_)
        
        self.update_showonly_true(to_scatter, all_unchecked=all_unchecked)
    
    def filter_pred_class(self, states_by_class):
        all_unchecked = ( 0 == sum(states_by_class.values()) )

        to_scatter = set()
        for class_, state in states_by_class.items():
            if state or all_unchecked:
                to_scatter.add(class_)
        
        self.update_showonly_pred(to_scatter, all_unchecked=all_unchecked)

    def update_showonly_true(self, classes, all_unchecked):
        """
        Hide all other label but class_

        :param classes: labels (decoded) to search and plot
        """

        logging.info("begin hiding...")

        for i in self.ax.get_children():
            if isinstance(i, matplotlib.collections.PathCollection):
                i.remove()

        for class_ in classes:
            similars = self.proj_by_class[class_]
            similar_idxs = self.index_by_class[class_]

            similars_good = [idx for idx in similar_idxs if idx in self.index_good_predicted ]
            similars_bad  = [idx for idx in similar_idxs if idx in self.index_bad_predicted ]
            if len(similars_bad):
                if all_unchecked and str(class_) == str(self.special_class):
                    continue
                self.ax.scatter(x=np.array([self.proj[i] for i in similars_bad])[:, 0],
                                y=np.array([self.proj[i] for i in similars_bad])[:, 1],
                                color='r',
                                marker='+')
            if len(similars_good):
                if all_unchecked and class_ == self.special_class:
                    continue
                self.ax.scatter(x=np.array([self.proj[i] for i in similars_good])[:, 0],
                                y=np.array([self.proj[i] for i in similars_good])[:, 1],
                                color='b',
                                marker='+')
        if all_unchecked:
            self.ax.scatter(
                    x=np.array([self.proj[i] for i in self.index_by_class[self.special_class]])[:,0],
                    y=np.array([self.proj[i] for i in self.index_by_class[self.special_class]])[:,1],
                    color='g',
                    marker='.')
            self.ax.set_title(self.ax_base_title)
        else:
            self.ax.set_title(''.join([str(class_)+' ' for class_ in classes]))

        self.refresh_graph()

    def update_showonly_pred(self, classes, all_unchecked):
        """
        Hide all other label but class_

        :param classes: labels (decoded) to search and plot
        """

        logging.info("begin hiding...")

        for i in self.ax.get_children():
            if isinstance(i, matplotlib.collections.PathCollection):
                i.remove()

        for class_ in classes:
            similars = self.proj_by_class[class_]
            similar_idxs = [idx for idx, this_class in enumerate(self.y_pred_decoded) if this_class==class_]

            similars_good = [idx for idx in similar_idxs if idx in self.index_good_predicted ]
            similars_bad  = [idx for idx in similar_idxs if idx in self.index_bad_predicted ]
            if len(similars_bad):
                if all_unchecked and str(class_) == str(self.special_class):
                    continue
                self.ax.scatter(x=np.array([self.proj[i] for i in similars_bad])[:, 0],
                                y=np.array([self.proj[i] for i in similars_bad])[:, 1],
                                color='r',
                                marker='+')
            if len(similars_good):
                if all_unchecked and class_ == self.special_class:
                    continue
                self.ax.scatter(x=np.array([self.proj[i] for i in similars_good])[:, 0],
                                y=np.array([self.proj[i] for i in similars_good])[:, 1],
                                color='b',
                                marker='+')
        if all_unchecked:
            self.ax.scatter(
                    x=np.array([self.proj[i] for i in self.index_by_class[self.special_class]])[:,0],
                    y=np.array([self.proj[i] for i in self.index_by_class[self.special_class]])[:,1],
                    color='g',
                    marker='x')
            self.ax.set_title(self.ax_base_title)
        else:
            self.ax.set_title(''.join([str(class_)+' ' for class_ in classes]))

        self.refresh_graph()

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

    def find_similar_clusters(self, x_g, y_g,
                              similarity_check=contains_dominant,
                              propagation='proximity'):
        """
        Find the coordinates of all similar clusters

        A cluster is considered similar if similarity_check function returns True
        Not all clusters are checked depending on the :param propagation: parameter

        :param propagation: set to "proximity" or "all"
                - proximity means that only (recursively) adjacent tiles are checked
                - all means that every tile is checked
        :param similarity_check: function that finds if two tiles are "similar"
                                 it should returns True in this case, input are (x0y0, xy)
                                 ..seealso:: contains_dominant
        :type x_g: int
        :type y_g: int
        """
        similar_clusters = []

        if propagation == 'proximity':
            _, similar_clusters = self.proximity_search(
                (x_g, y_g),
                (x_g, y_g),
                set(),
                set(),
                similarity_check
            )
        elif propagation == 'all':
            similar_clusters = self.exhaustive_search(
                (x_g, y_g),
                similarity_check
            )

        return similar_clusters

    def exhaustive_search(self, x0y0, similarity_check):
        """
        Search ALL tiles and check if similar

        :return: array of tiles similar to original according to similarity_check

        :param x0y0: original tile to compare others to
        :param similarity_check: function returning True if tiles are "similar"
        """
        x_0, y_0 = x0y0

        similar_clusters = []
        grid_axis_iterator = range(
            int(-self.resolution / 2) - 1,
            int(self.resolution / 2) + 1
        )

        for x_g, y_g in itertools.product(grid_axis_iterator, grid_axis_iterator):
            if similarity_check(self, (x_0, y_0), (x_g, y_g)):
                similar_clusters.append((x_g, y_g))

        return similar_clusters

    def proximity_search(self, x0y0, xy, already_checked, similars, similarity_check): #TODO
        """
        Recursive function that check if (recursively) adjacent tiles are similar

        Function starts its search at x0y0 (the cluster to which it compares others
        Then it uses :param similarity_check: function to find if similars exist in its neighbors
        proximity_search is then called again recursively on its similar neighbors

        :param similarity_check: the function that finds if two tiles are "similar"
        :param already_check: the set of already checked tiles
        :param similars: the set of similar tiles within the already_checked' ones
        :param x0y0: is the original tile from which we compare new ones
        """

        x0, y0 = x0y0
        x, y = xy

        if (x, y) in already_checked:
            return already_checked, similars

        already_checked.add((x, y))

        if similarity_check(self, (x, y), (x0, y0)):

            similars.add((x, y))

            if x + self.size_centroid < self.amplitude:
                already_checked, similars = self.proximity_search(
                        (x0, y0),
                        (x + self.size_centroid, y),
                        already_checked,
                        similars,
                        similarity_check
                        )
            if x - self.size_centroid > -self.amplitude:
                already_checked, similars = self.proximity_search((x0, y0), (x - 1, y),
                                                                  already_checked,
                                                                  similars,
                                                                  similarity_check)
            if y + 1 < self.resolution / 2:
                already_checked, similars = self.proximity_search((x0, y0), (x, y + 1),
                                                                  already_checked,
                                                                  similars,
                                                                  similarity_check)
            if y - 1 > -self.resolution / 2:
                already_checked, similars = self.proximity_search((x0, y0), (x, y - 1),
                                                                  already_checked,
                                                                  similars,
                                                                  similarity_check)
        return already_checked, similars

    def filter_class(self, states_by_class):
        all_unchecked = ( 0 == sum(states_by_class.values()) )

        to_scatter = set()
        for class_, state in states_by_class.items():
            if state or all_unchecked:
                to_scatter.add(class_)
        
        self.update_showonly(to_scatter, all_unchecked=all_unchecked)


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

        self.summary_axe.clear()
        self.summary_axe.axis('off')

        if button == 1:
            clicked_cluster = self.clusterizer.predict([(x,y)])[0]

            self.delimit_cluster(clicked_cluster, color=self.manual_cluster_color)
            self.update_summary(clicked_cluster)

            self.print_summary(self.summary_axe)

        elif button == 3:
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
        for i in [
                *self.ax.get_children(),
                *self.heat_entropy.get_children(),
                *self.heat_proportion.get_children(),
                ]:
            if isinstance(i, matplotlib.collections.PathCollection):
                i.remove()
            elif isinstance(i, matplotlib.lines.Line2D):
                if i.get_color() == self.manual_cluster_color:
                    i.remove()
        
        logging.info("scatterplot: drawing observations")
        self.ax.scatter(
            self.x_proj_good[:, 0],
            self.x_proj_good[:, 1],
            color='b', marker="+"
        )
        self.ax.scatter(
            self.x_proj_bad[:, 0],
            self.x_proj_bad[:, 1],
            color='r',
            marker='+'
        )
        self.ax.scatter(
            self.x_proj_null[:, 0],
            self.x_proj_null[:, 1],
            marker='x',
            color='g'
        )
        logging.info("scatterplot: ready")

    def reset_summary(self):
        """
        Reset the local summary
        """
        self.local_effectif = {}
        self.local_proportion = {}
        self.local_confusion_by_class = { class_:{ class__:0  for class__ in self.labels } for class_ in self.labels }
        self.local_bad_count_by_class = { class_:0 for class_ in self.labels }
        self.local_classes = set()
        self.local_sum = 0
        self.currently_selected_cluster = []

    def label_mesh(self):
        """
        Labels the mesh centroids

        Useful for heatmap right after this method. Should be called just
        after a change in clustering method. Parameter is implicitely the
        clusteriser of the vizualization
        """

        self.cluster_by_idx = self.clusterizer.predict(self.proj)
        all_cluster_labels = set(self.cluster_by_idx)
        
        index_by_label = { label:[] for label in all_cluster_labels }

        class_by_cluster = {
                label:{
                    class_:0 for class_ in self.proj_by_class }
                for label in all_cluster_labels
                }
        
        cluster_null = dict()
        cluster_good = dict()
        cluster_bad  = dict()
        cluster_good_count_by_class = dict()
        cluster_bad_count_by_class  = dict()
        
        for idx, label in enumerate(self.cluster_by_idx):
            index_by_label[label].append(idx)
            class_by_cluster[label][self.y_true_decoded[idx]]+=1
           

        logging.info('clustering: analyze each one')
        for label in all_cluster_labels:
            count = len(index_by_label[label])

            cluster_good[label] = 0
            cluster_bad[label]  = 0
            cluster_null[label] = 0
            cluster_good_count_by_class[label]={}
            cluster_bad_count_by_class[label]={}

            for i in index_by_label[label]:
                if i in self.index_good_predicted:
                    cluster_good[label]+=1
                    try:
                        cluster_good_count_by_class[label][self.y_true_decoded[i]]+=1
                    except KeyError:
                        cluster_good_count_by_class[label][self.y_true_decoded[i]]=1
                else:
                    cluster_bad[label]+=1
                    try:
                        cluster_bad_count_by_class[label][self.y_true_decoded[i]]+=1
                    except KeyError:
                        cluster_bad_count_by_class[label][self.y_true_decoded[i]]=1
                if i in self.index_not_predicted:
                    cluster_null[label]+=1
        
        logging.info('labelling: mesh centroids')
        centroids_label = self.clusterizer.predict(self.mesh_centroids)
        logging.info('labelling: done')

        self.centroids_label       = centroids_label
        self.cluster_good_count    = cluster_good
        self.cluster_bad_count     = cluster_bad
        self.cluster_good_count_by_class    = cluster_good_count_by_class
        self.cluster_bad_count_by_class     = cluster_bad_count_by_class
        self.cluster_null_count    = cluster_null
        self.index_by_label        = index_by_label
        self.class_by_cluster      = class_by_cluster

    def delimit_cluster(self, cluster, **kwargs):
        """
        Delimits one cluster by drawing lines around it
        """
        size = len(self.centroids_label)
        borders = set()
    
        for idx, xy in enumerate(self.mesh_centroids):
            if self.centroids_label[idx] == cluster:
                label_down_neighbor = self.centroids_label[max(idx-self.resolution,0)]
                label_left_neighbor = self.centroids_label[max(idx-1,0)]
                label_right_neighbor = self.centroids_label[min(idx+1,size-1)]
                label_up_neighbor = self.centroids_label[min(idx+self.resolution,size-1)]
                
                x, y = xy

                if label_down_neighbor != cluster:
                    for axe in self.axes_needing_borders:
                          axe.add_artist(
                              matplotlib.lines.Line2D(
                                  xdata = (
                                      x-self.size_centroid/2,
                                      x+self.size_centroid/2),
                                  ydata = (
                                      y-self.size_centroid/2,),
                                  **kwargs,
                                  )
                              )
                if label_up_neighbor != cluster:
                    for axe in self.axes_needing_borders:
                          axe.add_artist(
                              matplotlib.lines.Line2D(
                                  xdata = (
                                      x-self.size_centroid/2,
                                      x+self.size_centroid/2),
                                  ydata = (
                                      y+self.size_centroid/2,),
                                  **kwargs,
                                  )
                              )
                if label_left_neighbor != cluster:
                    for axe in self.axes_needing_borders:
                          axe.add_artist(
                              matplotlib.lines.Line2D(
                                  xdata = (
                                      x-self.size_centroid/2,),
                                  ydata = (
                                      y-self.size_centroid/2,
                                      y+self.size_centroid/2,),
                                  **kwargs,
                                  )
                              )
                if label_right_neighbor != cluster:
                    for axe in self.axes_needing_borders:
                          axe.add_artist(
                              matplotlib.lines.Line2D(
                                  xdata = (
                                      x+self.size_centroid/2,),
                                  ydata = (
                                      y-self.size_centroid/2,
                                      y+self.size_centroid/2,),
                                  **kwargs,
                                  )
                              )
        self.refresh_graph()
    
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
        for idx,xy in enumerate(self.mesh_centroids):

            current_centroid_label = self.centroids_label[idx]
            x, y = xy[0], xy[1]
            try:
                label_down_neighbor = self.centroids_label[idx-self.resolution]
                if label_down_neighbor != current_centroid_label:
                    try:
                        frontier[(label_down_neighbor, current_centroid_label)]
                    except KeyError:
                        current_frontier = frontier_builder(
                                    self.class_by_cluster[label_down_neighbor],
                                    self.class_by_cluster[current_centroid_label]
                                    )
                        if current_frontier > -np.inf:
                            frontier[(label_down_neighbor, current_centroid_label)] = current_frontier
            except KeyError:
                pass
            
            try:
                label_left_neighbor = self.centroids_label[idx-1]
                if label_left_neighbor != current_centroid_label:
                    try:
                        frontier[(label_left_neighbor, current_centroid_label)]
                    except KeyError:
                        current_frontier = frontier_builder(
                                    self.class_by_cluster[label_left_neighbor],
                                    self.class_by_cluster[current_centroid_label]
                                    )
                        if current_frontier > -np.inf:
                            frontier[(label_left_neighbor, current_centroid_label)] = current_frontier
            except KeyError:
                pass

        frontier = { key:frontier[key] for key in frontier if frontier[key] != -np.inf }
        
        if normalize_frontier:
            max_frontier = frontier[max(frontier, key=frontier.get)]
            min_frontier = frontier[min(frontier, key=frontier.get)]

            frontier_amplitude = max_frontier - min_frontier
            
            if frontier_amplitude:
                frontier = { key:frontier[key]-min_frontier / frontier_amplitude for key in frontier }

        logging.info('borders: cleaning')
        for axe in axes:
            for i in axe.get_children():
                if isinstance(i, plt.Line2D):
                    i.remove()

        logging.info('borders: drawing')
        for idx,xy in enumerate(self.mesh_centroids):

            current_centroid_label = self.centroids_label[idx]
            x, y = xy[0], xy[1]

            #if x+size_rect>0>x-size_rect and y+size_rect>0>y-size_rect:ipdb.set_trace()
            try:
                label_down_neighbor = self.centroids_label[idx-self.resolution]
                if label_down_neighbor != current_centroid_label:
                    frontier_density = frontier[(label_down_neighbor, current_centroid_label)]
                    for axe in axes:
                        axe.add_artist(
                            matplotlib.lines.Line2D(
                                xdata = (
                                    x-self.size_centroid/2,
                                    x+self.size_centroid/2),
                                ydata = (
                                    y-self.size_centroid/2,),
                                color='black',
                                alpha= 1 - frontier_density,
                                )
                            )
            except KeyError:
                pass

            try:
                label_left_neighbor = self.centroids_label[idx-1]
                if label_left_neighbor != current_centroid_label:
                    frontier_density = frontier[(label_left_neighbor, current_centroid_label)]
                    for axe in axes:
                        axe.add_artist(
                            matplotlib.lines.Line2D(
                                ydata = (
                                    y-self.size_centroid/2,
                                    y+self.size_centroid/2),
                                xdata = (
                                    x-self.size_centroid/2,),
                                color='black',
                                alpha=1 - frontier_density,
                                )
                            )
            except KeyError:
                pass
        logging.info('borders: ready')

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
        centroid_label = {}
        logging.info('heatmap: drawing proportion heatmap')

        for idx,xy in enumerate(self.mesh_centroids):

            current_centroid_label = self.centroids_label[idx]
            x, y = xy[0], xy[1]
            count = (
                    self.cluster_good_count.get(current_centroid_label, 0)
                    +self.cluster_bad_count.get(current_centroid_label, 0)
                    )

            if count:
                proportion_correct = self.cluster_good_count[current_centroid_label] / float(count)
                proportion_null    = self.cluster_null_count[current_centroid_label] / float(count)
                proportion_incorrect = 1 - proportion_correct
            else:
                proportion_correct = 1
                proportion_null = 1
                proportion_incorrect = 1

            red   = proportion_incorrect
            green = proportion_null
            blue  = proportion_correct

            all_colors[self.resolution - int(((idx-idx%self.resolution)/self.resolution))-1][idx%self.resolution] = [red, green, blue]


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
        centroid_label = {}
        logging.info('heatmap entropy: drawing')
        
        entropys = []

        for idx,xy in enumerate(self.mesh_centroids):

            current_centroid_label = self.centroids_label[idx]
            x, y = xy[0], xy[1]
            current_entropy = 0
            
            try:
                if len(self.index_by_label[current_centroid_label]) == 0:
                    current_entropy = 0
                else:
                    current_entropy = (
                        cross_entropy(
                            self.total_individual,
                            self.class_by_cluster[current_centroid_label]
                            )
                        )
            except KeyError:
                current_entropy = 0 # cluster does not exist -> empty dummy cluster
            entropys.append(current_entropy)

        min_entropys = min(entropys)
        max_entropys = max(entropys)
        amplitude_entropys = max_entropys - min_entropys
        logging.info('heatmap entropy: max cross-entropy='+str(max_entropys)+' min='+str(min_entropys))

        for idx, xy in enumerate(self.mesh_centroids):
            try:
                current_entropy = entropys[idx]
            except IndexError:
                current_entropy = min_entropys

            normalized_entropy = ((current_entropy - min_entropys) / amplitude_entropys)
            x, y = xy[0], xy[1]
            
            # all_colors[idx%self.resolution].append(normalized_entropy)
            all_colors[self.resolution - int(((idx-idx%self.resolution)/self.resolution))-1][idx%self.resolution] = normalized_entropy


            #logging.debug('entropy here is '+str(self.entropys[idx])+' and color coef '+str(coef))
            
        logging.info('heatmap entropy: done')
        return all_colors


    def request_new_frontiers(self, method):
        """
        Init new frontiers based on a similarity mesure between clusters
        """
        method = method.lower()
        logging.info('frontiers : requiring new delimitations '+method)

        if method == 'bhattacharyya':
            logging.debug('frontiers: set up to '+method)
            self.similarity_measure = bhattacharyya
            self.normalize_frontier=True
        elif method =='all':
            self.similarity_measure = lambda x,y:0
            self.normalize_frontier=False
            logging.debug('frontiers: set up to '+method)
        elif method == 'none':
            self.similarity_measure = lambda x,y:1
            self.normalize_frontier=False
            logging.debug('frontiers: set up to '+method)
            logging.info('frontiers : applied '+method)
            return
        
        self.apply_borders(
                self.normalize_frontier,
                self.similarity_measure,
                self.axes_needing_borders
                )
        self.refresh_graph()
        logging.info('frontiers : applied '+method)

    def request_new_clustering(self, method):
        """
        Init and fit a new clustering engin, then update the heatmaps

        :param method: clustering engine to use ..seealso:clustering module
        """
        method = method.lower()
        logging.info("cluster: requesting a new "+method+" engine")
        if method is None:
            method=self.last_clusterizer_method
        if method=='kmeans':
            self.clusterizer = clustering.KmeansClusterizer(
                    n_clusters=self.n_clusters,
                    )
        elif method=='dbscan':
            self.clusterizer = clustering.DBSCANClusterizer()
        else:
            self.clusterizer = clustering.DummyClusterizer(
                    resolution=self.resolution,
                    )

        self.last_clusterizer_method = method
        self.clusterizer.fit(xs=self.proj)
        logging.info("cluster: done")

        self.label_mesh()
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
        
        self.refresh_graph()

    def update_summary(self, current_cluster):
        """
        Add the data of cluster (:param x_g:, :param y_g:) to the local-tobeplotted summary

        Three objects are important inside the object Vizualization and need to be updated :
            - self.currently_selected_cluster is the collection of selected tiles
            - self.local_classes contains labels inside current_cluster
            - self.local_effectif contains the effetif of each label inside current_cluster
            - self.local_sum the sum of local_effectif
            - self.local_proportion is the ratio of good/total predicted inside cluster, by label

        :param current_cluster: cluster name selected by click
        """
        to_include = self.class_by_cluster[current_cluster]
        to_include = { k:to_include[k] for k in to_include if to_include[k]!=0 }

        if current_cluster in self.currently_selected_cluster:
            return
        else:
            self.currently_selected_cluster.append(current_cluster)

        new_rows = set(to_include.keys()) - self.local_classes

        logging.info("Classes already detected :" + str(self.local_classes))
        logging.info("Classes detected on new click :" + str(set(to_include.keys())))
        logging.info("Classes to add to summary :" + str(set(new_rows)))

        rows_to_update = self.local_classes.intersection(set(to_include.keys()))
        self.local_classes = self.local_classes.union(set(to_include.keys()))
        self.local_sum = sum(to_include.values()) + self.local_sum

        for c in new_rows:
            self.local_effectif[c] = to_include[c]
            self.local_proportion[c] = self.cluster_good_count_by_class[current_cluster].get(c,0) / (self.cluster_bad_count_by_class[current_cluster].get(c,0) + self.cluster_good_count_by_class[current_cluster].get(c,0))
        for current_cluster in self.currently_selected_cluster:
            for idx in self.index_by_label[current_cluster]:
                if idx in self.index_bad_predicted:
                    current_class = self.y_true_decoded[idx]
                    self.local_bad_count_by_class[current_class] += 1
                    self.local_confusion_by_class[current_class][self.y_pred_decoded[idx]]+=1

        self.local_confusion_by_class_sorted = { k:[] for k in self.local_confusion_by_class.keys() }
        for class_, errors in self.local_confusion_by_class.items():
            self.local_confusion_by_class_sorted[class_] = Counter(errors).most_common(2)

        for c in rows_to_update:
            self.local_proportion[c] = (
                (
                    self.local_proportion[c] * self.local_effectif[c]
                    + self.cluster_good_count_by_class[current_cluster].get(c,0) / (self.cluster_bad_count_by_class[current_cluster].get(c,0)+self.cluster_good_count_by_class[current_cluster].get(c,0)) * to_include.get(c, 0)
                ) / (self.local_effectif[c] + to_include.get(c, 0))
            )
            self.local_effectif[c] += self.cluster_good_count_by_class[current_cluster].get(c,0)+self.cluster_bad_count_by_class[current_cluster].get(c,0)

    def get_selected_indexes(self):
        """
        Find indexes of xs in selected clusters
        """
        indexes_selected = []
        for label in self.currently_selected_cluster:
            for idx in self.index_by_label[label]:
                indexes_selected.append(idx) 

        return indexes_selected

    def print_summary(self, axe, max_row=15):
        """
        Print a short summary with basic stats (occurrence, classification rate) on an axe

        This method gets its data mainly from self.local_proportion and self.local_effectif,
        these objects are self-updated when needed and contain the data of the user-selected clusters
        
        :param max_row: max number of row to add in table summary
        :param axe: the matplotlib axe in which the stats will be plotted
        """

        row_labels = list(self.local_classes)

        values = [
            [
                (
                    '{0:.0f}'.format(self.local_effectif[c])+ "  ("
                    + '{0:.2f}'.format(self.local_effectif[c]/self.total_individual[c]*100)+"%)"),
                (
                    '{0:.2f}'.format(self.local_proportion[c]*100)+"% ("+
                    '{0:.2f}'.format((self.local_proportion[c]-self.proportion_by_class[c])*100)+"%)"
                    ),
                (
                    '{0:.0f}'.format(self.total_individual[c])+' ('+
                    '{0:.2f}'.format(self.total_individual[c]/float(len(self.proj))*100)+'%)'
                    ),
                

                (
                    ' '.join([
                        str(class_mistaken)[:6]+' ('+'{0:.0f}'.format(
                            error_count/float(self.local_bad_count_by_class[c])*100
                            )
                        +'%) '
                        for class_mistaken, error_count in self.local_confusion_by_class_sorted[c] if self.local_bad_count_by_class[c]!=0
                        ])
                    ),
                    
            ]
            for c in row_labels
        ]

        arg_sort = np.argsort([self.local_effectif[c] for c in row_labels])

        values = [values[i] for i in arg_sort[::-1]]
        row_labels = [row_labels[i][:6] for i in arg_sort[::-1]]

        # add row "all" for recap :

        max_row    = min(max_row, min(len(values), len(row_labels)))-1
        values     = values[:max_row]
        row_labels = row_labels[:max_row]
        
        values.append([self.local_sum, .856789, len(self.proj), ' '])
        row_labels.append('all')

        self.rows = row_labels

        summary = axe.table(
            cellText=values,
            rowLabels=row_labels,
            colLabels=self.cols,
            loc='center',
        )

        summary.auto_set_font_size(False)
        summary.set_fontsize(8)
        logging.info("Details=loaded")

        self.refresh_graph()

    def print_global_summary(self, ax, max_row=9):
        
        cols = ['accuracy', 'effectif']
        most_common_classes = Counter(
                {
                    c:len(self.index_by_class[c]) for c in self.labels
                    }
                ).most_common(max_row)

        row_labels = np.array(most_common_classes)[:,0]
        values = [
            [
                '{0:.2f}'.format(self.proportion_by_class[c]*100)+"%",
                (
                    '{0:.0f}'.format(self.total_individual[c])+' ('+
                    '{0:.2f}'.format(self.total_individual[c]/float(len(self.proj))*100)+'%)'
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

    def export(self, output_path):
        logging.info('exporting:...')
        pd.DataFrame(
                [
                    self.x_raw[idx]
                    for idx,c in enumerate(self.cluster_by_idx)
                    if c in self.currently_selected_cluster
                    ]
                ).to_csv(output_path)
        logging.info('exporting: done')
    
    def view_details_figure(self):
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

        gs=GridSpec(3,4)
        
        #self.view_details = View_details(self.x_raw)
        self.viz_handler = Viz_handler(self, self.main_fig, self.onclick)
        
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
        self.heat_proportion = self.main_fig.add_subplot(gs[1,3], sharex=self.ax, sharey=self.ax)

        self.heat_entropy = self.main_fig.add_subplot(gs[0,3], sharex=self.ax, sharey=self.ax)
        self.heat_entropy.set_title('\nHeatmap: cross-entropy cluster/all')
        self.heat_entropy.axis('off')

        self.axes_needing_borders = (self.ax, self.heat_proportion, self.heat_entropy)

        # draw heatmap
        logging.info("heatmap=calculating")
        '''
        self.heatmaps = []
        self.add_heatmap(self.heatmap_proportion, self.heat_proportion)
        self.add_heatmap(self.heatmap_entropy, self.heat_entropy)
        '''
        self.heatmaps = []
        self.add_heatmap(
                self.heatmap_proportion,
                self.heat_proportion,
                title='Heatmap: proportion correct predictions')
        self.add_heatmap(
                self.heatmap_entropy,
                self.heat_entropy,
                title='Heatmap: cross-entropy Cluster-All')
       
        self.label_mesh()
        
        self.update_all_heatmaps()
        logging.info("heatmap=ready")

        # draw scatter plot
        self.reset_viz()
        self.request_new_frontiers('none')

        logging.info('Vizualization=ready')

    def show(self):
        self.viz_handler.show()

    def refresh_graph(self):
        self.viz_handler.refresh()
