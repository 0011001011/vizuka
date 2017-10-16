import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import random

import logging
import wordcloud
from collections import Counter

from vizuka.graphics import qt_helpers
from vizuka.graphics import drawing
from vizuka.cluster_viewer import make_plotter


CLUSTER_PLOTTER = {}
            
class Cluster_viewer(matplotlib.figure.Figure):

    def __init__(self, features_to_display, x_raw, x_raw_columns, show_dichotomy=True):
        super().__init__()
        self.x_raw = x_raw
        self.x_raw_columns = x_raw_columns
        self.show_dichotomy = show_dichotomy # deprecated

        self.features_to_display = features_to_display
        self.spec_by_name = {}
        self.cluster_view_selected_indexes = []

        self.spec = gridspec.GridSpec(
                len(features_to_display.keys()),
                2, wspace=0.2
                )
        
        for idx,feature_name in enumerate(features_to_display.keys()):
            for plotter in features_to_display[feature_name]:
                self.spec_by_name[feature_name+plotter] = {}
                self.spec_by_name[feature_name+plotter]['good'] = self.spec[idx%2]
                self.spec_by_name[feature_name+plotter]['bad' ] = self.spec[idx%2+1]

                if plotter not in CLUSTER_PLOTTER.keys():
                    CLUSTER_PLOTTER[plotter] = make_plotter(plotter)
       

    def update_cluster_view(self, clicked_cluster, index_by_cluster_label, indexes_good, indexes_bad):
        """
        Updates the axes with the data of the clicked cluster

        clicked cluster: the label of the cluster you clicked
        index_by_cluster_label: indexs of datas indexed by cluster label (set containing int)
        indexes_good: indexes of all good predictions
        indexes_bad: indexes of all bad predicitons
        """
        self.clear()

        self.cluster_view_selected_indexes += index_by_cluster_label[clicked_cluster]

        selected_xs_raw  ={'all': [self.x_raw[idx] for idx in self.cluster_view_selected_indexes]}
        if self.show_dichotomy:
            selected_xs_raw['good'] = [self.x_raw[idx] for idx in self.cluster_view_selected_indexes if idx in indexes_good]
            selected_xs_raw['bad' ] = [self.x_raw[idx] for idx in self.cluster_view_selected_indexes if idx in indexes_bad ]
        
        columns_to_display = [list(self.x_raw_columns).index(i) for i in self.features_to_display]
        data_to_display = {
                'all':
                        {
                        self.x_raw_columns[i]:[x[i] for x in selected_xs_raw['all']]
                        for i in columns_to_display
                        }
                    }
        if self.show_dichotomy:
            data_to_display['good'] = {
                self.x_raw_columns[i]:[x[i] for x in selected_xs_raw['good']]
                for i in columns_to_display
                }
            data_to_display['bad'] = {
                self.x_raw_columns[i]:[x[i] for x in selected_xs_raw['bad']]
                for i in columns_to_display
                }

        def plot_it(plotter, data_to_display, data_name, fig, spec_to_update_, key):

            spec_to_update = spec_to_update_[key]
            data = data_to_display[key][data_name]
            import ipdb
            ipdb.set_trace()
            axe = plotter(data, fig, spec_to_update)
            if 'log' in data_to_display[key][data_name]:
                data_name += ' - log'
            data_name +=  ' - {} predictions'.format(key)

            if axe:
                axe.set_title(data_name)


        for key in ['good', 'bad']:
            for data_name in self.features_to_display:
                for plotter_name in self.features_to_display[data_name]:
                    plotter = CLUSTER_PLOTTER[plotter_name]
                    spec_to_update = self.spec_by_name[data_name+plotter_name]
                    plot_it(plotter, data_to_display, data_name, self, spec_to_update, key) 
        
    def reset(self):
        self.clear()
        self.cluster_view_selected_indexes = []

def moar_filters(
        window,
        right_dock,
        features,
        all_features_categories,
        features_to_filter,
        viz_engine,
        ):

    """
    Adds requested filters to the given window.
    Links that to tha vizuka display in the viz_engine
    """

    raw_inputs = features
    raw_inputs_columns = all_features_categories
    features_name = features_to_filter
    
    def get_feature_col(feature_name):
        return list(raw_inputs_columns).index(feature_name)

    features = {
            feature_name:set() for feature_name in features_name
            }
    indexes_by_feature_by_feature_name = {
            feature_name:set() for feature_name in features_name
            }

    # go through raw_inputs and list all possible features
    # for the ones in self.features_name
    for idx, input_ in enumerate(raw_inputs):
        for feature_name in features_name:
            feature = input_[ get_feature_col(feature_name) ]

            features[feature_name].add(feature)

    action = {}
    for feature_name in features:
        feature_col = get_feature_col(feature_name)
        action[feature_name] = lambda x:viz_engine.filter_by_feature(feature_col, x)
        logging.info('registering filter {} ({})'.format(feature_name, feature_col))
        qt_helpers.add_checkboxes(
                window,
                feature_name,
                features[feature_name],
                action[feature_name],
                right_dock,
                checked_by_default=True,
                )
