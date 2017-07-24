import labelling
import clustering
import dim_reduction
from config.references import (
    DATA_PATH,
    VERSION,
    REDUCTION_SIZE_FACTOR,
    TSNE_DATA_PATH,
    PARAMS,
    MODEL_PATH,
    DO_CALCULUS,
)

# from config.references import *
"""


"""
import logging
import math
import itertools
from collections import Counter

import numpy as np
import keras
import matplotlib
matplotlib.use('Qt4Agg')  # noqa
from matplotlib import pyplot as plt
from matplotlib import patches
from PyQt4 import QtGui
from PyQt4.QtCore import Qt
import ipdb
import pandas as pd

"""
from shared_helpers import config
DATA_VIZ_CONFIG = config.load_config(__package__)
"""

def rgb_to_hex(red, green, blue):
    """
    Convert (int, int, int) RGB to hex RGB
    """

    return '#%02x%02x%02x' % (red, green, blue)


def entropy(my_dic):
    """
    StraightForward entropy calculation

    :param my_dict: dict of occurence of different classes
    :return: discrete entropy calculation
    """
    effectif_total = sum(my_dic.values())
    s = 0
    for effectif in my_dic.values():
        proportion = effectif / effectif_total
        if proportion > 0:
            s += proportion * math.log(proportion)
    return -s

def cross_entropy(dict1, dict2):
    """
    Cross-entropy between two dicts
    dict1 must contains all keys in dict2
    """
    sum_dict1, sum_dict2 = sum(dict1.values()), sum(dict2.values())
    ce = 0

    for label in dict2:
       ce -= dict2[label]/sum_dict2*math.log(dict1[label]/float(sum_dict1))

    return ce

def bhattacharyya(dict1, dict2):
    """
    Similarity measure between two distribution
    
    :param dict1: dict with key:class, value:nb of observations
    """
    s = 0
    for i in {*dict1, *dict2}:
        s+=(dict1.get(i,0)*dict2.get(i,0))**.5
    return -math.log(s) if s!=0 else -np.inf

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

def separate_prediction(y_pred_decoded, y_true_decoded, name_of_void):  # TODO
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


def make_grids(proj,
               y_true_decoded,
               index_good_predicted,
               index_bad_predicted,
               index_not_predicted,
               resolution,
               ):
    """
    Build grids with proportions of bad/good/not predicted/proportion goodbad etc...
    A grid is a matrix which contains agregated data, resolution is its number of row (or col)

    :param proj: t-SNE projected data
    :param y_true_decoded: true labels, decoded (human-readable)
    :param index_good_predicted: ..seealso separate_predictions
    :param index_bad_predicted:  ..seealso separate_predictions
    :param index_not_predicted:  ..seealso separate_predictions
    :param resolution: size of grid (max number of square by row/column)

    :return:
        grid_good       : data well classified
        grid_bad        : data bad classified
        grid_total      : all data
        grid_proportion : score of prediction as a float (1:all good, 0:all bad)
        grid_sum        : number of points in grid_total
    """

    p_amp = find_amplitude(proj)

    grid_axis_iterator = range(
        int(-resolution / 2) - 1,
        int(resolution / 2) + 1
    )

    # grid_good contains
    grid_good = {i: {j: {} for j in grid_axis_iterator} for i in grid_axis_iterator}
    grid_bad = {i: {j: {} for j in grid_axis_iterator} for i in grid_axis_iterator}
    grid_null = {i: {j: 0 for j in grid_axis_iterator} for i in grid_axis_iterator}
    grid_total = {i: {j: {} for j in grid_axis_iterator} for i in grid_axis_iterator}

    grid_proportion = {i: {j: {} for j in grid_axis_iterator} for i in grid_axis_iterator}
    grid_null_proportion = {i: {j: {} for j in grid_axis_iterator} for i in grid_axis_iterator}

    grid_sum = {i: {j: 0 for j in grid_axis_iterator} for i in grid_axis_iterator}

    for idx in index_good_predicted:
        x = proj[idx]
        x1, x2 = x[0], x[1]
        z1, z2 = find_grid_position(x1, x2, resolution, p_amp)

        true_name = y_true_decoded[idx]

        grid_good[z1][z2][true_name] = grid_good[z1][z2].get(true_name, 0) + 1
        grid_total[z1][z2][true_name] = grid_total[z1][z2].get(true_name, 0) + 1

        grid_sum[z1][z2] += 1
    
    logging.info("grid_good=ready")

    for idx in index_bad_predicted:
        x = proj[idx]
        x1, x2 = x[0], x[1]
        z1, z2 = find_grid_position(x1, x2, resolution, p_amp)

        true_name = y_true_decoded[idx]

        grid_bad[z1][z2][true_name] = grid_bad[z1][z2].get(true_name, 0) + 1
        grid_total[z1][z2][true_name] = grid_total[z1][z2].get(true_name, 0) + 1

        grid_sum[z1][z2] += 1
    logging.info("grid_bad=ready")

    for idx in index_not_predicted:
        x = proj[idx]
        x1, x2 = x[0], x[1]
        z1, z2 = find_grid_position(x1, x2, resolution, p_amp)
        grid_null[z1][z2] += 1
    logging.info("grid_null=ready")

    ###################
    # Make the heatmap
    #
    # 0 is all false 1 is all true

    for x in grid_proportion:
        for y in grid_proportion[x]:
            if grid_total[x][y] == {}:
                grid_proportion[x][y] = -1
                grid_null_proportion[x][y] = -1
            else:
                if grid_bad[x][y] == {}:
                    grid_proportion[x][y] = 0
                else:
                    b = 0
                    # cette boucle est moche mais sum(vide) => bug
                    for a in grid_bad[x][y].values():
                        b += a
                    grid_proportion[x][y] = b / float(grid_sum[x][y])

                if grid_null[x][y] == 0:
                    grid_null_proportion[x][y] = 0
                else:

                    grid_null_proportion[x][y] = grid_null[x][y] / float(grid_sum[x][y])
    logging.info("grid_proportion=ready")
    logging.info("grid_null_proportion=ready")

    return (
        grid_bad,
        grid_good,
        grid_null,
        grid_total,
        grid_proportion, grid_null_proportion,
        grid_sum,
    )


def show_occurences_total(x, y, grid, resolution, amplitude):
    """
    Finds key associated with the largest value in the grid fragment containing (x, y)
    Use it with visualization.grid_total to get the most frequent label

    :param resolution: size of grid (number of square by row/column)
    :param amplitude:  size of embedded space, as max of axis (rounded as an int)
                       e.g: [[-1,1],[.1,.1] as an amplitude of 2
    :type x: float
    :type y: float
    :return: key of biggest value in grid
    """

    z1, z2 = find_grid_position(x, y, resolution, amplitude)
    if z1 not in grid.keys():
        return "out of space"
    elif z2 not in grid[z1].keys():
        return "out of space"
    elif len(grid[z1][z2]) == 0:
        return "nothing found here"
    else:
        return max(grid[z1][z2], key=grid[z1][z2].get)


def dist(a, b):
    """
    Euclidian distance
    """
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**.5


def find_nearest(x, y, all_data):
    """
    Finds the nearest point in all_data (not optimized!)

    :type all_data: array( (float,float) )
    :return: (x_nearest,y_nearest), index_nearest
    :rtype: (float, float), int
    """

    best_distance = float('inf')
    best_point = None
    idx = 0

    logging.info("searching nearest point to", x, ":", y)

    for i, point in enumerate(all_data):
        new_dist = dist([x, y], point)
        if best_distance > new_dist:
            best_point = point
            best_distance = new_dist
            idx = i
    logging.info("nearest seems to be ", best_point,
                 " at distance ", best_distance)

    return point, idx


def find_similar(account_encoded, y_true_decoded, embedded):
    """
    Get indexes in proj of all similar point
    Similar points are data with same labels as :param account_encoded:

    :param account_encoded: label of initial point
    :param y_true_decoded:  real labels decoded
    :param embedded:        euclidian space

    :rtype: array((x_similar, y_similar)), array(indexes)
    """

    idxs = []
    similars = []

    logging.info("similar to "+str(account_encoded.argsort()[-1]))

    for i, account in enumerate(y_true_decoded[:len(embedded)]):
        if account == account_encoded:
            similars.append(embedded[i])
            idxs.append(i)
    logging.info("found ", len(similars), " similars points")

    return np.array(similars), idxs

def find_projected_in_cluster(cluster, cluster_by_idx):
    """
    Returns raw x in cluster containing (param:x,param:y)
    """
    selected_idx = set()

    for idx, current_cluster in enumerate(cluster_by_idx):
        if cluster == current_cluster:
            selected_idx.add(idx)

    return selected_idx

def find_projected_in_rect(x, y, proj, resolution, amplitude, transactions_raw):
    """
    Returns raw transactions in tile containing (param:x,param:y)
    """

    x_g, y_g = find_grid_position(x, y, resolution, amplitude)
    transactions_index = []

    for idx, (x_p, y_p) in enumerate(proj):
        if (x_g, y_g) == find_grid_position(x_p, y_p, resolution, amplitude):
            transactions_index.append(idx)

    transactions = [transactions_raw[idx] for idx in transactions_index]

    return transactions


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
            special_class=0,
            class_decoder=(lambda x: x), class_encoder=(lambda x: x),
            output_path='output.csv',
            ):
        """
        Central function, draw heatmap + scatter plot + zoom + annotations on tSNE data

        :param y_pred: vector of predicted labels
        :param y_true: vector of true labels
        :param proj: vector of t-SNE projected data
        :param class_decoder: function to decode labels machinetohuman
        :param class_encoder: dict to encode labels humantomachine
        """

        logging.info("Vizualization=generating")

        self.manual_cluster_color = 'purple'
        self.output_path = output_path

        self.y_pred = y_pred
        self.y_true = y_true
        self.y_true_decoded = [class_decoder(y) for y in self.y_true]
        self.y_pred_decoded = [class_decoder(y) for y in self.y_pred]
        self.proj = proj
        self.x_raw = x_raw

        self.proj_by_id = {y: [] for y in self.y_true_decoded}

        for idx, projection in enumerate(self.proj):
            self.proj_by_id[self.y_true_decoded[idx]].append(projection)

        # convert dict values to np.array
        for class_ in self.proj_by_id:
            self.proj_by_id[class_] = np.array(self.proj_by_id[class_])

        self.resolution = resolution # counts of tiles per row/column
        self.class_decoder = class_decoder

        self.shift_held = False
        self.ctrl_held = False

        self.cols = ['effectif local', 'proportion local',
                     'effectif global', 'proportion global']
        self.local_effectif = {}
        self.local_proportion = {}
        self.local_classes = set()
        self.local_sum = 0
        self.currently_selected_cluster = []
        self.cursor_ids = [0]

        # Get the real labels found in true y
        self.labels = set(self.class_decoder(self.y_true[i]) for i in range(len(self.y_true)))

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

        
        logging.info("grid=griding")
        (
            self.grid_bad,
            self.grid_good,
            self.grid_null,
            self.grid_total,
            self.grid_proportion,
            self.grid_null_proportion,
            self.grid_sum,

        ) = make_grids(

            self.proj,
            self.y_true_decoded,
            self.index_good_predicted,
            self.index_bad_predicted,
            self.index_not_predicted,
            self.resolution,
        )


        # TODO put this monstruosity in make_grids
        self.total_individual = {}
        self.grid_proportion_individual_global = {a: 0 for a in self.labels}
        self.grid_proportion_individual = {
            k: {k2: {} for k2 in self.grid_total[k]} for k in self.grid_total
        }

        for account in self.labels:
            self.total_individual[account] = 0
            for x in self.grid_total:
                for y in self.grid_total[x]:
                    self.total_individual[account] += self.grid_total[x][y].get(account, 0)
                    try:
                        self.grid_proportion_individual[x][y][account] = (
                            self.grid_good[x][y].get(account, 0)
                            / self.grid_total[x][y].get(account, 0)
                        )
                    except ZeroDivisionError:
                        self.grid_proportion_individual[x][y][account] = -1

            self.grid_proportion_individual_global[account] = 0
            sum_account = 0
            for x in self.grid_proportion_individual:
                for y in self.grid_proportion_individual[x]:
                    self.grid_proportion_individual_global[account] += (
                        self.grid_proportion_individual[x][y].get(account, 1)
                        * self.grid_total[x][y].get(account, 0)
                    )

                    sum_account += self.grid_total[x][y].get(account, 0)

            try:
                self.grid_proportion_individual_global[account] /= sum_account
            except ZeroDivisionError:
                self.grid_proportion_individual_global[account] = -1

        logging.info("grid=ready")
        # Sort good/bad/not predictions in t-SNE space
        logging.info("projections=sorting")
        self.x_proj_good = np.array([self.proj[i] for i in self.index_good_predicted])
        self.x_proj_bad  = np.array([self.proj[i] for i in self.index_bad_predicted])
        self.x_proj_null = np.array([self.proj[i] for i in self.index_not_predicted])
        logging.info("projections=ready")
        
        #self.clusterizer = clustering.DummyClusterizer(resolution=self.resolution)
        logging.info('clustering engine=fitting')
        self.clusterizer = clustering.KmeansClusterizer()
        self.clusterizer.fit(self.proj)
        logging.info('clustering engine=ready')
        #self.similarity_measure = lambda x,y:0
        self.similarity_measure = bhattacharyya
        self.normalize_frontier = True
        

    #######################################
    # Similarity functions to draw clusters

    def get_dominant(self, x_g, y_g):
        """
        Returns dominant class of cluster
        """
        if self.grid_total[x_g][y_g] == {}:
            return None
        else:
            return max(self.grid_total[x_g][y_g], key=self.grid_total[x_g][y_g].get)

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

    def find_specific_clusters(self, class_):
        """
        Finds all clusters which contain :param class_:

        :param class_: label (decoded) to look for
        """
        grid_axis_iterator = range(
            int(-self.resolution / 2) - 1,
            int(self.resolution / 2) + 1
        )

        selected_clusters = set()
        for i, j in itertools.product(grid_axis_iterator, grid_axis_iterator):
            if self.grid_total[i][j].get(class_, 0) > 0:
                selected_clusters.add((i, j))

        return selected_clusters

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

    def update_showonly(self, class_):
        """
        Hide all other label but class_

        :param class_: label (decoded) to search and plot
        """

        logging.info("begin hiding...")
        self.ax.clear()

        similars = self.proj_by_id[class_]
        self.ax.scatter(x=similars[:, 0],
                        y=similars[:, 1],
                        color='g',
                        marker='+')
        logging.info("done")

        plt.draw()

    def update_showall(self, class_, color="green"):
        """
        Colorizes on label with specific color

        :param class_: label (decoded) to colorize
        :param color: color to use for :param class_:
        """

        logging.info("begin colorizing...")
        similars = self.proj_by_id[class_]

        self.ax.scatter(
            similars[:, 0],
            similars[:, 1],
            color=color,
            marker='+'
        )

        # similarity_check = contains_dominant if not self.ctrl_held else lambda x:x
        similar_clusters = self.find_specific_clusters(class_=class_)

        for x_g, y_g in similar_clusters:
            clicked_cluster = self.clusterizer.predict([(x,y)])[0] #TODO SOON
            logging.info("colorizing cluster", x_g, y_g)
            self.update_summary(clicked_cluster)
            self.ax.add_patch(self.colorize_rect(x_g, y_g))

        self.print_summary(self.summary_axe)
        plt.draw()
        logging.info("done")

    def add_menulist(self, menu_name, button_name, categories, onlaunch):
        """
        Add a menu list with action button

        :param menu_name: the name of the list (displayed)
        :param button_name: the name of the button
        :param categories: categories available for selection
        :param onlaunch: action to trigger on click to button
        """

        root = self.f.canvas.manager.window
        panel = QtGui.QWidget()
        hbox = QtGui.QHBoxLayout(panel)
        
        class MenuList(QtGui.QListWidget):

            def __init__(self, categories):
                QtGui.QListWidget.__init__(self)
                self.add_items(categories)
                self.itemClicked.connect(self.item_click)
                self.selected = categories

            def add_items(self, categories):
                for category in categories:
                    item = QtGui.QListWidgetItem(category)
                    self.addItem(item)

            def item_click(self, item):
                self.selected = str(item.text())
        
        menulist = MenuList(categories)

        hbox.addWidget(menulist)
        launchButton = QtGui.QPushButton(button_name)
        launchButton.clicked.connect(lambda: onlaunch(menulist.selected))
        hbox.addWidget(launchButton)
        panel.setLayout(hbox)
        
        dock = QtGui.QDockWidget(menu_name, root)
        root.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.setWidget(panel)

        return menulist

    def add_button(self, name, action):
        """
        Adds a simple button
        """
        root = self.f.canvas.manager.window
        panel = QtGui.QWidget()
        hbox = QtGui.QHBoxLayout(panel)

        button = QtGui.QPushButton(name)
        button.clicked.connect(action)
        hbox.addWidget(button)
        panel.setLayout(hbox)

        dock = QtGui.QDockWidget(name, root)
        root.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.setWidget(panel)

    def add_text_panel(self, name, update):
        """
        Adds a text panel (how surprising) and binds it to a function

        :param name: name of Widget
        :param update: function to bind returnPressed event of textpanel
        """

        root = self.f.canvas.manager.window
        panel = QtGui.QWidget()
        hbox = QtGui.QHBoxLayout(panel)
        textbox = QtGui.QLineEdit(parent=panel)

        textbox.returnPressed.connect(update)
        hbox.addWidget(textbox)
        panel.setLayout(hbox)

        dock = QtGui.QDockWidget(name, root)
        root.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.setWidget(panel)

        return textbox

    def textbox_function_showonly(self):
        """
        Wrapper for textbox, to use self.update_showonly
        without specifying parameters
        """
        class_str = self.textboxs['show_only'].text()
        if class_str == '':
            self.reset_viz()
        else:
            class_ = int(class_str)
            self.update_showonly(class_)

    def textbox_function_showall(self):
        """
        Wrapper for textbox, to use self.update_showall
        without specifying parameters
        """
        class_str = self.textboxs['show_all'].text()
        if class_str == '':
            self.reset_viz()
        else:
            class_ = int(class_str)
            self.update_showall(class_)

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

    '''
    def onmotion(self, event):
        """
        What we do when the cursor moves
        (spoiler : nothing)
        """
        
        x = event.xdata
        y = event.ydata
        
        if x is not None and y is not None:

            x_g, y_g = find_grid_position(x,y, self.resolution,self.amplitude)
            graphs = [ self.ax ]
            
            
            i = len(self.ax.patches)

            for graph, (cursor_idx,cursor_id) in zip(graphs, enumerate(self.cursor_ids)):

                rect = self.colorize_rect(x_g, y_g)

                if self.cursor_ids != [0,0,0]:
                    while i>0:
                        if id(graph.patches[i-1]) == cursor_id:
                            del graph.patches[i-1]
                            i=0

                graph.add_patch(rect)
                self.cursor_ids[cursor_idx] = id(rect)

            logging.info("cursor position:", (x_g,y_g))
        
        plt.draw()
    '''

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

        x_g, y_g = find_grid_position(
            x,
            y,
            self.resolution,
            self.amplitude
        )

        if button == 1:

            # show dominant account in grid fragment, in title
            # select nearest point to mouse
            # colorize them all

            # find nearest point to click

            if self.shift_held:

                nearest, idx = find_nearest(x, y, self.proj)
                class_nearest = self.y_true_decoded[idx]
                logging.info("looking for class", class_nearest)
                similars, _ = find_similar(
                    class_nearest, self.y_true_decoded, self.proj)

                self.ax.scatter(
                    self.x_proj_good[:, 0],
                    self.x_proj_good[:, 1],
                    color='b',
                    marker="+"
                )
                self.ax.scatter(
                    self.x_proj_bad[:, 0],
                    self.x_proj_bad[:, 1],
                    color='r',
                    marker='+'
                )
                self.ax.scatter(
                    similars[:, 0],
                    similars[:, 1],
                    color='green',
                    marker='+'
                )
                dominant = str(
                    show_occurences_total(
                        x,
                        y,
                        self.grid_total,
                        self.resolution,
                        self.amplitude
                    )
                )
                self.ax.set_title(''.join([
                    'dominant class: ',
                    dominant,
                    ', colorizing ',
                    str(self.labels[idx])
                ]))
            

            clicked_cluster = self.clusterizer.predict([(x,y)])[0]

            self.delimit_cluster(clicked_cluster, color=self.manual_cluster_color)
            self.update_summary(clicked_cluster)

            self.ax.add_patch(self.colorize_rect(x_g, y_g))
            self.print_summary(self.summary_axe)

            selected_x_idx = find_projected_in_cluster(
                clicked_cluster,
                self.cluster_by_idx,
            )

            """
            logging.info('\n\n' + ('-' * 12) + '\nSelected transactions:')
            for idx in selected_x_idx:
                logging.info(self.x_raw[idx])
            """

            logging.debug('x=%s y=%s x_grid=%s y_grid=%s\n', x, y, x_g, y_g)

        elif button == 2:
            
            dominant = self.get_dominant(x_g, y_g)
            if dominant is None: return

            similars = self.proj_by_id[dominant]
            self.ax.scatter(
                similars[:, 0],
                similars[:, 1],
                color='green',
                marker='+'
            )

            propagation = 'all' if not self.shift_held else 'proximity'
            similar_clusters = self.find_similar_clusters(
                x_g,
                y_g,
                propagation=propagation
            )

            for x_g, y_g in similar_clusters:
                self.update_summary(x_g, y_g)
                self.ax.add_patch(self.colorize_rect(x_g, y_g))
            self.print_summary(self.summary_axe)

        elif button == 3:

            # reboot vizualization
            self.reset_summary()
            self.reset_viz()

        plt.draw()

    def reset_viz(self):
        """
        Reset (graphically) the vizualization
        ..note:: does not touch the summary array, for this use self.reset_summary()
        """
        logging.info("scatterplot: removing specific objects")
        for i in self.ax.get_children():
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
        self.local_classes = set()
        self.local_sum = 0
        self.currently_selected_cluster = []

    def colorize_rect(self, x_g, y_g):
        """
        Prepare empty rectangle adapted to tile to be used as a cursor
        """

        size_rect = self.amplitude / self.resolution
        zoom_rect = patches.Rectangle(
            (
                ((x_g / self.resolution)) * self.amplitude,
                (y_g / self.resolution) * self.amplitude
            ),
            size_rect, size_rect,
            fill=False, edgecolor='magenta'
        )

        return zoom_rect

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
                    class_:0 for class_ in self.proj_by_id }
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
        plt.draw()
    
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

        all_patches = []
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

                red   = int(255 * (1-proportion_correct))
                green = int(255 * proportion_null)
                blue  = int(255 * proportion_correct)

                color = rgb_to_hex(red, green, blue)
                
                x_rect = x - self.size_centroid/2 
                y_rect = y - self.size_centroid/2
                
                all_patches.append(
                        patches.Rectangle(
                                (x_rect, y_rect),
                                self.size_centroid,
                                self.size_centroid,
                                color=color,
                                ),
                        )
        logging.info('heatmap: proportion done')
        return all_patches


        """
        for x in self.grid_proportion:
            for y in self.grid_proportion[x]:
                if self.grid_proportion[x][y] != -1:

                    red   = int(255 * self.grid_proportion[x][y])
                    green = int(255 * self.grid_null_proportion[x][y])
                    blue  = int(255 * (1 - self.grid_proportion[x][y]))

                    color = rgb_to_hex(red, green, blue)

                    x_rect = x * (self.amplitude / float(self.resolution))
                    y_rect = y * (self.amplitude / float(self.resolution))

                    all_patches.append(
                        patches.Rectangle(
                            (x_rect, y_rect),
                            self.amplitude / self.resolution,
                            self.amplitude / self.resolution,
                            color=color,
                        ),
                    )

        return all_patches
        """

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

        all_patches = []
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

        for idx, xy in enumerate(self.mesh_centroids):
            try:
                current_entropy = entropys[idx]
            except IndexError:
                current_entropy = min_entropys

            coef = int((current_entropy - min_entropys) / amplitude_entropys * 255)
            color = rgb_to_hex(coef, coef, coef)
            x, y = xy[0], xy[1]

            all_patches.append(
                patches.Rectangle(
                    (x-self.size_centroid/2, y-self.size_centroid/2),
                    self.size_centroid,
                    self.size_centroid,
                    color=color,
                ),
            #logging.debug('entropy here is '+str(self.entropys[idx])+' and color coef '+str(coef))
            )
        logging.info('heatmap entropy: done')
        return all_patches

    def heatmap_proportion_v2(self):
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


        """
        for x in self.grid_proportion:
            for y in self.grid_proportion[x]:
                if self.grid_proportion[x][y] != -1:

                    red   = int(255 * self.grid_proportion[x][y])
                    green = int(255 * self.grid_null_proportion[x][y])
                    blue  = int(255 * (1 - self.grid_proportion[x][y]))

                    color = rgb_to_hex(red, green, blue)

                    x_rect = x * (self.amplitude / float(self.resolution))
                    y_rect = y * (self.amplitude / float(self.resolution))

                    all_patches.append(
                        patches.Rectangle(
                            (x_rect, y_rect),
                            self.amplitude / self.resolution,
                            self.amplitude / self.resolution,
                            color=color,
                        ),
                    )

        return all_patches
        """

    def heatmap_entropy_v2(self):
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

    def heatmap_density(self):
        """
        Prepare the patches for a density heatmap

        This method is a  heatmap_builder returning a list of patches to be plotted somewhere
        The effectif of fullest tile is calculated and used as normalization parameter,
        The plot is actually a logplot as it is more eye-friendly


        ..seealso:: add_heatmap
        """

        all_patches = []

        max_sum = 0
        for x in self.grid_sum:
            for y in self.grid_sum[x]:
                if max_sum < self.grid_sum[x][y]:
                    max_sum = self.grid_sum[x][y]

        for x in self.grid_sum:
            for y in self.grid_sum[x]:
                current = self.grid_sum[x][y]
                coef = int(
                    255 - 255 * math.log(1 + 4 * current / max_sum) / math.log(5)
                )
                color = rgb_to_hex(coef, coef, coef)

                x_rect = x * (self.amplitude / float(self.resolution))
                y_rect = y * (self.amplitude / float(self.resolution))

                all_patches.append(
                    patches.Rectangle(
                        (x_rect, y_rect),
                        self.amplitude / self.resolution,
                        self.amplitude / self.resolution,
                        color=color,
                    ),
                )

        return all_patches

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
        
        self.apply_borders(
                self.normalize_frontier,
                self.similarity_measure,
                self.axes_needing_borders
                )
        plt.draw()
        logging.info('frontiers : applied '+method)

    def request_new_clustering(self, method):
        """
        Init and fit a new clustering engin, then update the heatmaps

        :param method: clustering engine to use ..seealso:clustering module
        """
        method = method.lower()
        logging.info("cluster: requesting a new "+method+" engine")
        if method=='kmeans':
            self.clusterizer = clustering.KmeansClusterizer(
                    )
        else:
            self.clusterizer = clustering.DummyClusterizer(
                    resolution=self.resolution
                    )
        self.clusterizer.fit(xs=self.proj)
        logging.info("cluster: done")

        self.label_mesh()
        self.update_all_heatmaps_v2()

        self.apply_borders(
                self.normalize_frontier,
                self.similarity_measure,
                self.axes_needing_borders) 
        logging.info('borders: done')

    def update_all_heatmaps(self):
        """
        Get all heatmaps registered by add_heatmap and draw them from scratch
        """
        for (heatmap_builder, axe) in self.heatmaps:
            axe.clear()
            
            all_patches = heatmap_builder()
            logging.info("heatmaps: adding patches to "+str(axe))
            for p in all_patches:
                axe.add_patch(p)
            
            logging.info("heatmaps: "+str(axe)+" ready")

            axe.set_xlim(-self.amplitude / 2, self.amplitude / 2)
            axe.set_ylim(-self.amplitude / 2, self.amplitude / 2)
        
        plt.draw()

    def update_all_heatmaps_v2(self):
        """
        Get all heatmaps registered by add_heatmap and draw them from scratch
        """
        for (heatmap_builder, axe) in self.heatmaps_v2:
            axe.clear()
            
            heatmap_color = heatmap_builder()
            logging.info("heatmaps: adding patches to "+str(axe))
            axe.imshow(heatmap_color, interpolation='nearest', vmin=0, vmax=1, extent=(-self.amplitude-self.size_centroid/2, self.amplitude-self.size_centroid/2, -self.amplitude-self.size_centroid/2, self.amplitude-self.size_centroid/2), aspect='auto')
            
            logging.info("heatmaps: "+str(axe)+" ready")

            axe.set_xlim(-self.amplitude / 2, self.amplitude / 2)
            axe.set_ylim(-self.amplitude / 2, self.amplitude / 2)
        
        plt.draw()

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

        # print("grid_total keys:", self.grid_total.keys())
        # print("grid_total[0] keys:", self.grid_total[0].keys())
        
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

        #ipdb.set_trace()
        for c in new_rows:
            self.local_effectif[c] = to_include[c]
            self.local_proportion[c] = self.cluster_good_count_by_class[current_cluster].get(c,0) / (self.cluster_bad_count_by_class[current_cluster].get(c,0) + self.cluster_good_count_by_class[current_cluster].get(c,0))

        for c in rows_to_update:
            self.local_proportion[c] = (
                (
                    self.local_proportion[c] * self.local_effectif[c]
                    + self.cluster_good_count_by_class[current_cluster].get(c,0) / (self.cluster_bad_count_by_class[current_cluster].get(c,0)+self.cluster_good_count_by_class[current_cluster].get(c,0)) * to_include.get(c, 0)
                ) / (self.local_effectif[c] + to_include.get(c, 0))
            )
            self.local_effectif[c] += self.cluster_good_count_by_class[current_cluster].get(c,0)+self.cluster_bad_count_by_class[current_cluster].get(c,0)

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
                self.local_effectif[c],
                self.local_proportion[c],
                self.total_individual[c],
                self.grid_proportion_individual_global[c]
            ]
            for c in row_labels
        ]

        arg_sort = np.argsort(np.array(values)[:, 0])

        values = [values[i] for i in arg_sort[::-1]]
        row_labels = [row_labels[i] for i in arg_sort[::-1]]

        # add row "all" for recap :
        values.append([self.local_sum, .856789, len(self.proj), .856789])
        row_labels.append('all')

        axe.table(
            cellText=values[:max_row],
            rowLabels=row_labels[:max_row],
            colLabels=self.cols,
            loc='center'
        )
        #ipdb.set_trace()

        plt.draw()
    
    def add_heatmap_v2(self, heatmap_builder, axe):
        """
        Draw a heatmap based on a heatmap_builder on an axe

        :param heatmap_builder: a Vizualization parameterless method which returns patches
        :param axe: matplotlib axe object in which the heatmap will be plotted
        """

        self.heatmaps_v2.append((heatmap_builder, axe))

    def add_heatmap(self, heatmap_builder, axe):
        """
        Draw a heatmap based on a heatmap_builder on an axe

        :param heatmap_builder: a Vizualization parameterless method which returns patches
        :param axe: matplotlib axe object in which the heatmap will be plotted
        """

        self.heatmaps.append((heatmap_builder, axe))
    
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


    def plot(self):
        """
        Plot the Vizualization, define axes, add scatterplot, buttons, etc..
        """

        self.f = plt.figure()

        # main subplot with the scatter plot
        self.heat_proportion = plt.subplot(3, 1, (1, 2))
        self.heat_proportion.set_title('Heatmap: correctVSincorrect predictions')

        # summary_subplot with table of local stats
        self.summary_axe = plt.subplot(3, 2, 5)
        self.summary_axe.axis('off')

        # heatmap subplots
        # contain proportion of correct prediction and entropy
        self.ax = plt.subplot(3, 4, 11)
        self.ax.set_title('Observations')
        self.heat_entropy = plt.subplot(3, 4, 12)
        self.heat_entropy.set_title('Heatmap: cross-entropy localVSglobal')

        self.axes_needing_borders = (self.ax, self.heat_proportion, self.heat_entropy)

        def wrapper_show_occurences_total(x, y):
            return show_occurences_total(x, y, self.grid_total, self.resolution, self.amplitude)
        
        # draw heatmap
        logging.info("heatmap=calculating")
        '''
        self.heatmaps = []
        self.add_heatmap(self.heatmap_proportion, self.heat_proportion)
        self.add_heatmap(self.heatmap_entropy, self.heat_entropy)
        '''
        self.heatmaps_v2 = []
        self.add_heatmap_v2(self.heatmap_proportion_v2, self.heat_proportion)
        self.add_heatmap_v2(self.heatmap_entropy_v2, self.heat_entropy)
       
        self.label_mesh()
        
        self.update_all_heatmaps_v2()
        '''
        self.update_all_heatmaps()
        '''
        logging.info("heatmap=ready")

        # draw scatter plot
        self.reset_viz()
        
        # draw clusters borders
        self.apply_borders(
                self.normalize_frontier,
                self.similarity_measure,
                self.axes_needing_borders)

        # add mouse event
        logging.info("mouseEvents=adding")
        self.f.canvas.mpl_connect('button_press_event', self.onclick)
        self.f.canvas.mpl_connect('key_press_event', self.onmodifier_press)
        self.f.canvas.mpl_connect('key_release_event', self.onmodifier_release)
        logging.info("mouseEvents=ready")

        # add textbox
        self.textboxs = {}
        logging.info("textboxs=adding")
        self.textboxs['show_only'] = self.add_text_panel(
            'Show one label',
            self.textbox_function_showonly
        )
        self.textboxs['show_all'] = self.add_text_panel(
            'Select all with label',
            self.textbox_function_showall
        )
        logging.info("textboxs=ready")

        # add button
        self.add_button("Export x", lambda :self.export(self.output_path))

        # add menulist
        self.menulists = {}
        self.menulists['clustering_method'] = self.add_menulist(
                'Clustering method',
                'Clusterize', ['KMeans', 'Dummy'],
                self.request_new_clustering)
        self.menulists['clustering_method'] = self.add_menulist(
                'Borders',
                'Delimits',
                ['Bhattacharyya', 'All', 'None'],
                self.request_new_frontiers)

        logging.info('Vizualization=ready')


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    logging.info("Starting script")

    PARAMS['perplexities'] = [40, 50]
    PARAMS['learning_rates'] = [800, 1000]
    PARAMS['inits'] = ['random', 'pca']
    PARAMS['n_iters'] = [15000]

    logging.info("raw_data=loading")
    (
        x_small,
        y_small,
        class_encoder,
        class_decoder,

    ) = dim_reduction.load_raw_data()

    logging.info('raw_data=loaded')

    if DO_CALCULUS:
        logging.info("t-sne=learning")

        x_transformed, models = dim_reduction.learn_tSNE(
            PARAMS,
            VERSION,
            x_small,
            TSNE_DATA_PATH,
            REDUCTION_SIZE_FACTOR,
        )
        logging.info('t-sne=ready')
    else:
        logging.info("t-sne=loading")

        x_transformed, models = dim_reduction.load_tSNE(
            PARAMS,
            VERSION,
            TSNE_DATA_PATH,
            REDUCTION_SIZE_FACTOR,
        )
        logging.info('t-sne=ready')

    x_2D = x_transformed[(50, 1000, 'pca', 15000)]

    ###############
    # PREDICT

    if DO_CALCULUS:
        logging.info('RNpredictions=predicting')
        x_predicted = labelling.predict_rnn(
            x_small,
            y_small,
            path=MODEL_PATH,
            version=VERSION
        )
        logging.info('RNpredictions=ready')
    else:
        logging.info('RNpredictions=loading')
        x_predicted = labelling.load_predict(
            path=MODEL_PATH,
            version=VERSION
        )
        logging.info('RNpredictions=ready')
    logging.info("loading raw transactions for analysis..")

    transactions_raw = np.load(
        DATA_PATH + 'originals' + VERSION + '.npz'
    )['originals']

    f = Vizualization(
        x_raw = transactions_raw,
        proj=x_transformed[(50, 1000, 'pca', 15000)],
        y_true=y_small,
        y_pred=x_predicted,
        resolution=200,
        class_decoder=class_decoder,
        class_encoder=class_encoder,
    )

    f.plot()
    plt.show()
