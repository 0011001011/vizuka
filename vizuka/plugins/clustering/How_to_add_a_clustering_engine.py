###########################################################################################
#
# How to add a clustering engine ?
#
##################################
#
# Simply creates a class in a new module that
# implements vizuka.clustering.clusterizer.Clusterizer
#
# Below is a simple example, with kmeans.
# If you need special parameters, pass them in vizualization.py (search "make_clusterizer")
# If these special parameters are hyperparameters, use
# self.register_parameters (see below). This is important for Vizuka to
# know which parameters to ask the user when you launch the clustering
# engine from the qt interface.
#
#
##########################################################################################


###################
#
# WORKING EXAMPLE :
#
###################
#
#class KmeansClusterizer(Clusterizer):
#
#    def __init__(self, number_of_clusters=15):
#        """
#        Uses sklearn kmeans
#        """
#        self.register_parameters(
#                parameters={'number_of_clusters':number_of_clusters})
#
#        self.method = 'kmeans'
#        self.number_of_clusters = number_of_clusters
#
#        self.engine = KMeans(
#                n_clusters=int(float(self.number_of_clusters)),
#                )
#
#    def fit(self, xs):
#        """
#        Fit the datas and find clusterization adapted to the data provided
#
#        :param xs: data to clusterize
#        """
#        self.engine.fit(xs)
#
#    def predict(self, xs):
#        """
#        Predicts cluster label
#
#        :param xs: array-like of datas
#        :return:   list of cluster possible_outputs_list
#        """
#        return self.engine.predict(xs)
