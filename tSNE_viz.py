import dim_reduction
import labelling
from config.references import (
    DATA_PATH,
    VERSION,
    REDUCTION_SIZE_FACTOR,
    TSNE_DATA_PATH,
    PARAMS,
    MODEL_PATH,
    DEFAULT_RN,
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

    x_proj_amplitude = 1 + int(
        max(-min(np.array(proj)[:, 0]), max(np.array(proj)[:, 0])))
    y_proj_amplitude = 1 + int(
        max(-min(np.array(proj)[:, 1]), max(np.array(proj)[:, 1])))

    p_amp = 2 * max(x_proj_amplitude, y_proj_amplitude)

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

    logging.info("similar to ", account_encoded.argsort()[-1])

    for i, account in enumerate(y_true_decoded[:len(embedded)]):
        if account == account_encoded:
            similars.append(embedded[i])
            idxs.append(i)
    logging.info("found ", len(similars), " similars points")

    return np.array(similars), idxs


def find_projected_in_rect(x, y, proj, resolution, amplitude, transactions_raw):
    """
    Returns raw transactions in cluster containing (param:x,param:y)
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

    def __init__(self, y_pred, y_true, proj, resolution=100,
                 special_class=0,
                 class_decoder=(lambda x: x), class_encoder=(lambda x: x)):
        """
        Central function, draw heatmap + scatter plot + zoom + annotations on tSNE data

        :param y_pred: vector of predicted labels
        :param y_true: vector of true labels
        :param proj: vector of t-SNE projected data
        :param class_decoder: function to decode labels machinetohuman
        :param class_encoder: dict to encode labels humantomachine
        """

        logging.info("Vizualization=generating")

        self.y_pred = y_pred
        self.y_true = y_true
        self.y_true_decoded = [class_decoder(y) for y in self.y_true]
        self.y_pred_decoded = [class_decoder(y) for y in self.y_pred]
        self.proj = proj

        self.proj_by_id = {y: [] for y in self.y_true_decoded}

        for idx, projection in enumerate(self.proj):
            self.proj_by_id[self.y_true_decoded[idx]].append(projection)

        # convert dict values to np.array
        for class_ in self.proj_by_id:
            self.proj_by_id[class_] = np.array(self.proj_by_id[class_])

        self.resolution = resolution
        self.class_decoder = class_decoder

        self.shift_held = False
        self.ctrl_held = False

        self.cols = ['effectif local', 'proportion local',
                     'effectif global', 'proportion global']
        self.local_effectif = {}
        self.local_proportion = {}
        self.local_classes = set()
        self.local_sum = 0
        self.current_cluster = []
        self.cursor_ids = [0]

        # Get the real labels found in true y
        self.labels = set(self.class_decoder(self.y_true[i]) for i in range(len(self.y_true)))

        x_proj_amplitude = 1 + int(
            max(-min(np.array(self.proj)[:, 0]), max(np.array(self.proj)[:, 0])))
        y_proj_amplitude = 1 + int(
            max(-min(np.array(self.proj)[:, 1]), max(np.array(self.proj)[:, 1])))

        self.amplitude = 2 * max(x_proj_amplitude, y_proj_amplitude)

        index_bad_predicted, index_good_predicted, index_not_predicted = separate_prediction(
            self.y_pred_decoded,
            self.y_true_decoded,
            special_class
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
            index_good_predicted,
            index_bad_predicted,
            index_not_predicted,
            self.resolution,
        )
        # TODO put this monstruosity in make_grids
        self.grid_total_individual = {}
        self.grid_proportion_individual_global = {a: 0 for a in self.labels}
        self.grid_proportion_individual = {
            k: {k2: {} for k2 in self.grid_total[k]} for k in self.grid_total
        }

        for account in self.labels:
            self.grid_total_individual[account] = 0
            for x in self.grid_total:
                for y in self.grid_total[x]:
                    self.grid_total_individual[account] += self.grid_total[x][y].get(account, 0)
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
        self.x_proj_good = np.array([self.proj[i] for i in index_good_predicted])
        self.x_proj_bad = np.array([self.proj[i] for i in index_bad_predicted])
        self.x_proj_null = np.array([self.proj[i] for i in index_not_predicted])

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

    def proximity_search(self, x0y0, xy, already_checked, similars, similarity_check):
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

            if x + 1 < self.resolution / 2:
                already_checked, similars = self.proximity_search((x0, y0), (x + 1, y),
                                                                  already_checked,
                                                                  similars,
                                                                  similarity_check)
            if x - 1 > -self.resolution / 2:
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
            logging.info("colorizing cluster", x_g, y_g)
            self.update_summary(x_g, y_g)
            self.ax.add_patch(self.colorize_rect(x_g, y_g))

        self.print_summary(self.summary_axe)
        plt.draw()
        logging.info("done")

    def add_text_panel(self, name, update):
        """
        Adds a text panel (how surprising) and bind it to a function

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

            self.update_summary(x_g, y_g)
            self.ax.add_patch(self.colorize_rect(x_g, y_g))
            self.print_summary(self.summary_axe)

            selected_tx = find_projected_in_rect(
                x, y,
                self.proj,
                self.resolution,
                self.amplitude,
                transactions_raw
            )

            logging.info('\n\n' + ('-' * 12) + '\nSelected transactions:')
            for tx in selected_tx:
                logging.info(tx)

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
        self.ax.clear()
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

    def reset_summary(self):
        """
        Reset the local summary
        """
        self.local_effectif = {}
        self.local_proportion = {}
        self.local_classes = set()
        self.local_sum = 0
        self.current_cluster = []

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
        for x in self.grid_proportion:
            for y in self.grid_proportion[x]:
                if self.grid_proportion[x][y] != -1:

                    red = int(255 * self.grid_proportion[x][y])
                    green = int(255 * self.grid_null_proportion[x][y])
                    blue = int(255 * (1 - self.grid_proportion[x][y]))

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
        nb_class = len(self.grid_total_individual)
        normalization = -math.log(1 / nb_class)

        for x in self.grid_total:
            for y in self.grid_total[x]:
                if len(self.grid_total[x][y]) == 0:
                    current_entropy = 1
                else:
                    current_entropy = (
                        entropy(self.grid_total[x][y])
                        / normalization
                    )

                coef = int(
                    255 * min(
                        (
                            (
                                1
                                + math.log(
                                    current_entropy
                                    + math.exp(-1)
                                )
                            )
                            / math.log(
                                1
                                + math.exp(-1)
                            )
                        ),
                        1
                    )
                
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

    def update_summary(self, x_g, y_g):
        """
        Add the data of cluster (:param x_g:, :param y_g:) to the local-tobeplotted summary

        Three objects are important inside the object Vizualization and need to be updated :
            - self.current_cluster is the collection of selected tiles
            - self.local_classes contains labels inside current_cluster
            - self.local_effectif contains the effetif of each label inside current_cluster
            - self.local_sum the sum of local_effectif
            - self.local_proportion is the ratio of good/total predicted inside cluster, by label


        :param x_g: x_coordinate of tile
        :param y_g: y_coordinate of tile
        :type x_g: int
        :type y_g: int
        """

        # print("grid_total keys:", self.grid_total.keys())
        # print("grid_total[0] keys:", self.grid_total[0].keys())

        to_include = self.grid_total[x_g][y_g]

        if (x_g, y_g) in self.current_cluster:
            return
        else:
            self.current_cluster.append((x_g, y_g))

        new_rows = set(to_include.keys()) - self.local_classes

        logging.info("Classes already detected :" + str(self.local_classes))
        logging.info("Classes detected on new click :" + str(set(to_include.keys())))
        logging.info("Classes to add to summary :" + str(set(new_rows)))

        rows_to_update = self.local_classes.intersection(set(to_include.keys()))
        self.local_classes = self.local_classes.union(set(to_include.keys()))
        self.local_sum = sum(to_include.values()) + self.local_sum

        for c in new_rows:
            self.local_effectif[c] = to_include[c]
            self.local_proportion[c] = self.grid_proportion_individual[x_g][y_g][c]

        for c in rows_to_update:
            self.local_proportion[c] = (
                (
                    self.local_proportion[c] * self.local_effectif[c]
                    + self.grid_proportion_individual[x_g][y_g][c] * to_include.get(c, 0)
                ) / (self.local_effectif[c] + to_include.get(c, 0))
            )
            self.local_effectif[c] += self.grid_total[x_g][y_g].get(c, 0)

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
                self.grid_total_individual[c],
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

        plt.draw()

    def add_heatmap(self, heatmap_builder, axe):
        """
        Draw a heatmap based on a heatmap_builder on an axe

        :param heatmap_builder: a Vizualization parameterless method which returns patches
        :param axe: matplotlib axe object in which the heatmap will be plotted
        """

        all_patches = heatmap_builder()
        for p in all_patches:
            axe.add_patch(p)
        axe.set_xlim(-self.amplitude / 2, self.amplitude / 2)
        axe.set_ylim(-self.amplitude / 2, self.amplitude / 2)

    def plot(self):
        """
        Plot the Vizualization, define axes, add scatterplot, buttons, etc..
        """

        self.f = plt.figure()

        # main subplot with the scatter plot
        self.ax = plt.subplot(3, 1, (1, 2))

        # summary_subplot with table of local stats
        self.summary_axe = plt.subplot(3, 2, 5)
        self.summary_axe.axis('off')

        # heatmap subplots
        # contain proportion of correct prediction and entropy
        self.heat_proportion = plt.subplot(3, 4, 11)
        self.heat_entropy = plt.subplot(3, 4, 12)

        def wrapper_show_occurences_total(x, y):
            return show_occurences_total(x, y, self.grid_total, self.resolution, self.amplitude)
        
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

        # draw heatmap
        logging.info("heatmap=calculating")
        self.add_heatmap(self.heatmap_proportion, self.heat_proportion)
        self.add_heatmap(self.heatmap_entropy, self.heat_entropy)
        logging.info("heatmap=ready")

        # draw scatter plot
        self.reset_viz()

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
        proj=x_transformed[(50, 1000, 'pca', 15000)],
        y_true=y_small,
        y_pred=x_predicted,
        resolution=100,
        class_decoder=class_decoder,
        class_encoder=class_encoder,
    )

    f.plot()
    plt.show()
