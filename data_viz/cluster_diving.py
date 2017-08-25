import qt_handler
import matplotlib
from matplotlib import pyplot as plt
import logging
'''
from qt_handler import Qt_matplotlib_handler
class Cluster_diver(Qt_matplotlib_handler):

    def __init__(self, raw_inputs, raw_inputs_columns, features_name=[]):
        
        self.raw_inputs = raw_inputs
        self.raw_inputs_columns = raw_inputs_columns
        self.features_name = features_name

        super(Cluster_diver, self).__init__(self.main_fig)
        self.add_figure(self.main_fig, window=self.window)
        
        self.features = {
                feature_name:set() for feature_name in features_name
                }
        self.indexes_by_feature_by_feature_name = {
                feature_name:set() for feature_name in features_name
                }

        # go through raw_inputs and list all possible features
        # for the ones in self.features_name
        for idx, input_ in enumerate(raw_inputs):
            for feature_name in features_name:
                feature = input_[ self.get_feature_col(feature_name) ]

                self.features[feature_name].add(feature)
                """
                if feature not in self.indexes_by_feature_by_feature_name[feature_name]:
                    self.indexes_by_feature_by_feature_name[feature_name][feature]=set([idx])
                else:
                    self.indexes_by_feature_by_feature_name[feature_name][feature].add(idx)
                """

        for feature_name in self.features:
            self.add_checkboxes(
                    feature_name,
                    self.features[feature_name],
                    lambda x:self.viz_engine.filter_by_feature(get_feature_col(feature_name), x),
                    self.right_dock,
                    checked_by_default=True,
                    )

    def show(self):
        self.window.show()

    def get_corresponding_indexes(feature_name, feature):
        return self.indexes_by_feature_by_feature_name[feature_name][feature]

'''
def moar_filters(
        window,
        right_dock,
        features,
        features_categories,
        features_to_filter,
        viz_engine,
        ):

    raw_inputs = features
    raw_inputs_columns = features_categories
    features_name = features_to_filter
    
    def get_feature_col(feature_name):
        print('so({})={}'.format(feature_name, list(raw_inputs_columns).index(feature_name)))
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
            """
            if feature not in self.indexes_by_feature_by_feature_name[feature_name]:
                self.indexes_by_feature_by_feature_name[feature_name][feature]=set([idx])
            else:
                self.indexes_by_feature_by_feature_name[feature_name][feature].add(idx)
            """
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
