import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import logging

from vizuka import vizualization
from vizuka.clustering.clusterizer import Clusterizer


class DBSCANClusterizer(Clusterizer):

    def __init__(self, epsilon=1.5, min_samples=30):
        """
        Inits a DBSCAN clustering engine from sklearn
        Accepts the same arguments
        """
        self.epsilon     = float(epsilon)
        self.min_samples = int(float(min_samples))

        self.register_parameters(
                parameters={
                    'epsilon'    : self.epsilon,
                    'min_samples': self.min_samples}
                )

        self.method='dbscan'

        self.engine = DBSCAN(
                n_jobs      = 4,
                eps         = self.epsilon,
                min_samples = self.min_samples,
                )

    def fit(self, xs):
        """
        There is no dbscan.predict so...
        We are going to predict everything and
        put it on a big dict.

        This is stupid but thank sklearn for that.
        If you want to predict the class of a point
        not initially in your data (e.g the mesh_centroids)
        then the engine will first find the nearest fitted
        data, and give you its cluster labelling.

        :param xs: array-like of datas
        """
        xs_tuple = [ tuple(x) for x in xs ]
        tmp = self.engine.fit_predict(xs_tuple)
        
        self.predictions = {xs_tuple[idx]: predict for idx, predict in enumerate(tmp)}
        labels = set(tmp)

        def do(xs_tuple):
            tmp = self.engine.fit_predict(xs_tuple)
            self.predictions = {xs_tuple[idx]: predict for idx, predict in enumerate(tmp)}
            labels = set(tmp)

            f = plt.figure()
            s = f.add_subplot(111)
            s.set_title(str(len(labels))+" class")

            for i in labels:
                to_display = np.array([x for idx,x in enumerate(xs_tuple) if i == tmp[idx]])
                s.scatter(to_display[:,0], to_display[:,1])

            plt.show()
        
        # do(xs_tuple)

        self.kdtree = cKDTree(xs)
        self.xs = xs
        logging.info("DBSCAN found {} labels".format(len(labels)))

        # There is a problm here : all isolated points are classified -1
        # in DBSCAN, which is a problem for our interactive cluster selection
        # as selecting a title (labelled as the label of nearest point to its
        # "centroid") may lead to select all tiles labelled as -1 : this would
        # be very ugly

        class_min = min(labels)
        for key, class_ in self.predictions.items():
            if class_ <= -1:
                class_min-=1
                self.predictions[key] = class_min
        labels = set(self.predictions.values())
        
        logging.info("DBSCAN found {} labels after correction".format(len(labels)))

    def predict(self, xs):
        """
        Predicts cluster label
        :param xs: array-like of datas to classify
        ..seealso:: self.fit
        """
        current_predicts = []
        for x in xs:
            x_tuple = tuple(x)
            if x_tuple in self.predictions:
                current_predicts.append(self.predictions[x_tuple])
            else:
                current_predicts.append(
                    self.predictions[tuple(self.xs[self.kdtree.query(x)[1]])]
                )
        return current_predicts

