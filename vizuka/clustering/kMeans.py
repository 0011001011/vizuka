import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import cKDTree
import logging

from vizuka import vizualization
from vizuka.clustering.clusterizer import Clusterizer


class KmeansClusterizer(Clusterizer):

    required_arguments = ['Number of clusters']

    def __init__(self, required_arguments={'Number of clusters':15}):
        """
        Uses sklearn kmeans, accepts same arguments.
        Default nb of cluster : 120
        """
        self.engine = KMeans(
                n_clusters=required_arguments['Number of clusters'],
                )
        self.method='kmeans'

    def fit(self, xs):
        """
        Fit the datas and find clusterization adapted to the data provided

        :param xs: data to clusterize
        """
        self.engine.fit(xs)

    def predict(self, xs):
        """
        Predicts cluster label

        :param xs: array-like of datas
        :return:   list of cluster possible_outputs_list
        """
        return self.engine.predict(xs)
