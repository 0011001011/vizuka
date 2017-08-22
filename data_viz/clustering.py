'''
Clustering engine to use with Vizualization

If you want to implement one, dont forget to add it
on qt_handler, to be able to select on the IHM

3 methods are necessary to implement, cf Clusterizer():
    init    - set main params
    fit     - to prepare the algo for the data
    predict - to find out the cluster of a (x,y)
'''

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import KDTree
import ipdb

from . import vizualization


class Clusterizer():

        def __init__(self, *args, **kwargs):
            """
            Builds a clusterizer object e.g: kmeans
            Do not know the datas at this point, just pass it the bare
            minimum to initialize an engine.

            :param *args **kwargs: parameters passed to the clusterizer engine
                                   it can literally be whatever you want
            """
            self.engine = None

        def fit(self, xs):
            """
            First time the engine sees the data.
            Depending on the algorithm you may do your magic your own way
            and maybe store new variable in self, maybe store all the
            predicts for each x directly in a dict.

            :param xs: a list containing data to clusterize
            """
            pass


        def predict(self, xs):
            """
            Finds the cluster(s) in which the data is.

            :params xs: array-like of (x,y) points
            :return: array-like of cluster id
            """
            return (0, ) * len(xs)


class KmeansClusterizer(Clusterizer):

    def __init__(self, n_clusters=120, *args, **kwargs):
        """
        Uses sklearn kmeans, accepts same arguments.
        Default number of cluster : 120
        """
        self.engine = KMeans(n_clusters=n_clusters, *args, **kwargs)

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
        :return:   list of cluster labels
        """
        return self.engine.predict(xs)


class DBSCANClusterizer(Clusterizer):

    def __init__(self, *args, **kwargs):
        """
        Inits a DBSCAN clustering engine from sklearn
        Accepts the same arguments
        """
        self.engine = DBSCAN(n_jobs=4, *args, **kwargs)

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
        self.kdtree = KDTree(xs)
        self.xs = xs

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


class DummyClusterizer(Clusterizer):
    """
    The DummyClusterizer is a clustering engine which
    return the index of your point in a big mesh.

    Give it the resolution of your mesh and its amplitude,
    it will center it on (0,0) and "clusterize". There are
    resolution*resolution clusters, come of them being..
    hum.. empty yes.
    """

    def __init__(self, mesh):
        """
        Inits the "engine" by giving it a resolution.
        The resolution will be the square root of the
        number of clusters.
        """
        self.mesh   = mesh
        self.kdtree = KDTree(self.mesh)

    def fit(self, xs):
        """
        Fit to the data, for this it finds how big the mesh
        will need to be

        :param xs: array-like of data to clusterize
        """
        pass

    def predict(self, xs):
        """
        Simply give you the index of the mesh in which the
        data is, it is considered as a cluster label
        """
        return [self.kdtree.query(x)[1] for x in xs]


def make_clusterizer(xs, method='kmeans', **kwargs):
    """
    Clusterize the data with specified algorithm
    Naively assume you pass the right parameters for the right algo

    :param data:       array with shape (n,2) of inputs to clusterize
    :param method:     algo to use, supported: kmeans, dbscan, dummy
    :param n_clusters: number of clusters to find (if applicable)

    :return: a clusterizer object (instance of child of Clusterizer())
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
    -> UNUSED IN CODE BUT FANCY

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
    """
    Yes, I like to test my code with __main__
    """
    import dim_reduction as dr
    datas_sets, models = dr.load_tSNE()
    datas = datas_sets[50, 1000, 'pca', 15000]

    clusterizer = make_clusterizer(datas, method='kmeans', n_clusters=80)
    f, ax = plot_clusters(datas, clusterizer)

    plt.show()
