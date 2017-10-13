###########################################################################################
#
# How to add a clustering engine ?
#
##################################
#
# Simply creates a class in a new module that
# implements vizuka.clustering.clusterizer.Clusterizer
#
# Below is a simple example, with kmeans
# If you need special parameters, pass them in vizualization.py (search "make_clusterizer")
#
#
##########################################################################################


###################
#
# WORKING EXAMPLE :
#
###################
#
# from sklearn.cluster import KMeans
# from vizuka.clustering.clusterizer import Clusterizer
#
#
# class KmeansClusterizer(Clusterizer):
#
#     def __init__(self, n_clusters=120, *args, **kwargs):
#         """
#         Uses sklearn kmeans, accepts same arguments.
#         Default nb of cluster : 120
#         """
#         self.engine = KMeans(n_clusters=n_clusters, *args, **kwargs)
#         self.method='kmeans'
#
#     def fit(self, xs):
#         """
#         Fit the datas and find clusterization adapted to the data provided
#
#         :param xs: data to clusterize
#         """
#         self.engine.fit(xs)
#
#     def predict(self, xs):
#         """
#         Predicts cluster label
#
#         :param xs: array-like of datas
#         :return:   list of cluster possible_outputs_list
#         """
#         return self.engine.predict(xs)
