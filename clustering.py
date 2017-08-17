'''
Clustering engine to use with Vizualization

3 methods :
    init    - with different params
    fit     - to prepare the algo for the data
    predict - to find out the cluster of a (x,y)
'''

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import KDTree
import ipdb

from vizualization import find_amplitude, find_grid_positions


class Clusterizer():

        def __init__(self, *args, **kwargs):
            """
            Builds a clusterizer object e.g: kmeans

            :param *args **kwargs: parameters passed to the clusterizer engine
            """
            self.engine = None

        def predict(self, xs):
            """
            Gives the cluster in which the data is

            :params xys: array-like of (x,y) points
            :return: array-like of cluster id
            """
            return (0, ) * len(xs)


class KmeansClusterizer(Clusterizer):

    def __init__(self, n_clusters=120, *args, **kwargs):
        self.engine = KMeans(n_clusters=n_clusters, *args, **kwargs)

    def fit(self, xs):
        self.engine.fit(xs)

    def predict(self, xs):
        return self.engine.predict(xs)


class DBSCANClusterizer(Clusterizer):

    def __init__(self, *args, **kwargs):
        self.engine = DBSCAN(n_jobs=4, *args, **kwargs)

    def fit(self, xs):
        """
        There is no dbscan.predict so...
        """
        xs_tuple = [ tuple(x) for x in xs ]
        tmp = self.engine.fit_predict(xs_tuple)
        self.predictions = {xs_tuple[idx]: predict for idx, predict in enumerate(tmp)}
        self.kdtree = KDTree(xs)
        self.xs = xs

    def predict(self, xs):
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


class DummyClusterizer(Clusterizer):

    def __init__(self, resolution):
        self.resolution = resolution

    def fit(self, xs):
        self.amplitude = find_amplitude(xs)

    def predict(self, xys):

        attributed_cluster = []
        xgygs = find_grid_positions(xys, self.resolution, self.amplitude)

        attributed_cluster = [
            xgyg[0] + (xgyg[1] + self.resolution/2 + 2) * self.resolution
            for xgyg in xgygs
        ]

        return attributed_cluster


def make_clusterizer(xs, method='kmeans', **kwargs):
    """
    Clusterize the data with specified algorithm
    Naively assume you pass the right parameters for the right algo

    :param data: array with shape (n,2) of in put to clusterize
    :param method: algo to use, supported: kmeans
    :param n_clusters: number of clusters to find (if apply)
    """
    
    clusterizer = None
    if method == 'kmeans':
        clusterizer = KmeansClusterizer(kwargs['n_clusters'])
    elif method == 'dbscan':
        clusterizer = DBSCANClusterizer()
    else:
        clusterizer = DummyClusterizer(kwargs['resolution'])
    clusterizer.fit(xs)

    return clusterizer


def plot_clusters(data, clusterizer):
    """
    Nicely returns a viz of the clusters

    :param data: array with shape (n,2) of in put to clusterize
    :param clusterizer: ..seealso::clusterizer
    """

    h = .2     # point in the grid[x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels
    Z = clusterizer.predict(np.c_[xx.ravel(), yy.ravel()])
    ipdb.set_trace()

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    f, ax = plt.subplots()
    ax.imshow(Z, interpolation='nearest',
              extent=(xx.min(), xx.max(), yy.min(), yy.max()),
              cmap=plt.cm.Paired,
              aspect='auto', origin='lower')

    ax.plot(data[:, 0], data[:, 1], 'k.', markersize=2)
    
    """
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    """

    ax.set_title('K-means clustering')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    return f, ax


if __name__ == '__main__':
    import dim_reduction as dr
    datas_sets, models = dr.load_tSNE()
    datas = datas_sets[50, 1000, 'pca', 15000]

    clusterizer = make_clusterizer(datas, method='kmeans', n_clusters=80)
    f, ax = plot_clusters(datas, clusterizer)

    plt.show()
