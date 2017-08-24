from data_viz.qt_handler import Qt_matplotlib_handler

from matplotlib import pyplot as plt


class Cluster_diver(Qt_matplotlib_handler):

    def __init__(self, raw_inputs, features_name=[]):
        
        self.raw_inputs = raw_inputs
        main_fig = matplotlib.figure.Figure()
        super(Cluster_diver, self).__init__(main_fig)
        
        if self.features_name:
            self.raw_inputs['columns'].index(feature_name)
        self.features = {
                feature_name:set() for feature_name in features_name
                }
        self.indexes_by_feature_by_feature_name = {
                feature_name:set() for feature_name in features_name
                }


        for idx, input_ in enumerate(raw_inputs['originals']):
            for feature_name in features_name:
                feature = input_[ self.get_feature_col(feature_name) ]

                self.features[feature_name].add(feature)
                if feature not in self.indexes_by_feature_by_feature_name[feature_name]:
                    self.indexes_by_feature_by_feature_name[feature_name][feature]=set([idx])
                else:
                    self.indexes_by_feature_by_feature_name[feature_name][feature].add(idx)

        for feature_name in self.features:
            self.add_checkboxes(
                    feature_name,
                    self.features[feature_name],
                    lambda x:self.viz_engine.filter_by_feature(feature_name, x),
                    self.right_dock,
                    checked_by_default=True,
                    )

    def get_feature_col(self, feature_name):
        self.raw_inputs['columns'].index(feature_name)

    def get_corresponding_indexes(feature_name, feature):
        return self.indexes_by_feature_by_feature_name[feature_name][feature]
