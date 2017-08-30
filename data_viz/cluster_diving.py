import qt_handler
import matplotlib
from matplotlib import pyplot as plt
import logging
from dynamic_subplot import add_subplot

class Cluster_viewer(matplotlib.figure.Figure):

    def __init__(self, features_to_display):
        super().__init__()
        self.subplot_by_name = {}
        for feature_name in features_to_display.keys():
            for plotter in features_to_display[feature_name]:
                self.subplot_by_name[feature_name+plotter] = add_subplot(self)

    def clear(self):
        for subplot in self.subplot_by_name.values():
            subplot.clear()

'''
def add_cluster_view(
        window,
        right_dock,
        features,
        all_features_categories,
        features_to_display,
        viz_engine,
        ):
    """
    Adds the requested features to a new display in
    the given window. Links it to the viz_engine and
    updates on cluster selection
    """
    raw_inputs = features
    raw_inputs_columns = all_features_categories
    features_name = features_to_display

    figure = matplotlib.figure.Figure()
    
    for feature_name in features_name:
        pass
    add_figure(figure, window)
'''

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
