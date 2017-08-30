import qt_handler
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import logging
import wordcloud
from dynamic_subplot import add_subplot
from collections import Counter


def plot_density(data, axe, scale='linear'):

    data = [float(d) for d in data]
    bins = 100 # int(len(data)/10)
    hist, bins = np.histogram(data, bins=bins)
    width = .7 *(bins[1] - bins[0])
    center = (bins[:-1] + bins[1:])/2
    
    axe.set_yscale(scale)
    axe.bar(center, hist, align='center', width=width)

def plot_logdensity(data, axe):
    plot_density(data, axe, scale='log')

def plot_wordcloud(data, axe):
    # ok fuck let's be stupid for testing purpose
    data = [str(d) for d in data]
    words_freq = Counter(sum([phrase.split(' ') for phrase in data], []))
    wc = wordcloud.WordCloud()
    
    wc.fit_words(words_freq)
    wc.background_color = 'white'
    wc.scale=10 # for better resolution
    wc.relative_scaling = .5 # for sizing words not only on ranks but also reative freq
    axe.axis('off')
    axe.imshow(wc.to_array())

def plot_counter(data, axe):
    c = Counter(data)
    x = [l for l in c.keys()]
    y = [c[l] for l in x]

    graduation = np.linspace(0, len(y), len(y))
    axe.bar(height=y, left=graduation)
    axe.set_xticks(graduation)
    axe.set_xticklabels([str(i) for i in x])

CLUSTER_PLOTTER = {
        'logdensity': plot_logdensity,
        'density': plot_density,
        'wordcloud': plot_wordcloud,
        'counter': plot_counter,
        }

            
class Cluster_viewer(matplotlib.figure.Figure):

    def __init__(self, features_to_display, x_raw, x_raw_columns):
        super().__init__()
        self.x_raw = x_raw
        self.x_raw_columns = x_raw_columns

        self.features_to_display = features_to_display
        self.subplot_by_name = {}
        self.cluster_view_selected_indexes = []

        for feature_name in features_to_display.keys():
            for plotter in features_to_display[feature_name]:
                self.subplot_by_name[feature_name+plotter] = add_subplot(self)

    def clear(self):
        for subplot in self.subplot_by_name.values():
            subplot.clear()
        self.cluster_view_selected_indexes = []

    def update_cluster_view(self, clicked_cluster, index_by_cluster_label):
        """
        Updates the axes with the data of the clicked cluster
        """
        self.cluster_view_selected_indexes += index_by_cluster_label[clicked_cluster]
        selected_xs_raw  = [self.x_raw[idx] for idx in self.cluster_view_selected_indexes]
        
        columns_to_display = [list(self.x_raw_columns).index(i) for i in self.features_to_display]
        data_to_display = {
                self.x_raw_columns[i]:[x[i] for x in selected_xs_raw]
                for i in columns_to_display
                }

        for data_name in data_to_display:
            for plotter_name in self.features_to_display[data_name]:
                plotter = CLUSTER_PLOTTER[plotter_name]
                axe_to_update = self.subplot_by_name[data_name+plotter_name]
                axe_to_update.clear()
                plotter(data_to_display[data_name], axe_to_update)
                if 'log' in data_to_display[data_name]:
                    data_name += ' - log'
                axe_to_update.set_title(data_name)


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
    Links that to tha data_viz display in the viz_engine
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
        qt_handler.add_checkboxes(
                window,
                feature_name,
                features[feature_name],
                action[feature_name],
                right_dock,
                checked_by_default=True,
                )
