'''
Clustering engine to use with Vizualization

If you want to implement one, dont forget to add it
on qt_handler, to be able to select on the IHM

3 methods are necessary to implement, cf Clusterizer():
    init    - set main params
    fit     - to prepare the algo for the data
    predict - to find out the cluster of a (x,y)
'''
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import cKDTree
import logging

from vizuka import vizualization


def load_cluster(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
        


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
            self.method=''

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
        

        def save_cluster(self, path):
            with open(path, 'wb') as f:
                pickle.dump(self, f)


class KmeansClusterizer(Clusterizer):

    def __init__(self, n_clusters=120, *args, **kwargs):
        """
        Uses sklearn kmeans, accepts same arguments.
        Default nb of cluster : 120
        """
        self.engine = KMeans(n_clusters=n_clusters, *args, **kwargs)
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

class LoaderClusterizer(Clusterizer):

    def __init__(self):
        """
        Simply loads a npz with all labels
        """
        data = pickle.load(open('vizuka/data/models/clusterizer.pkl', 'rb'))
        self.xs, self.engine = data # self.engine is here a collection of labels
        self.kdtree = cKDTree(self.xs)

    def fit(self, xs):
        pass

    def predict(self, xs):
        """
        Return the predictions found in the predictions .pkl
        """
        return self.kdtree.query(xs)[1]
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
        """


class DBSCANClusterizer(Clusterizer):

    def __init__(self, *args, **kwargs):
        """
        Inits a DBSCAN clustering engine from sklearn
        Accepts the same arguments
        """
        self.engine = DBSCAN(n_jobs=4, eps=1.6, min_samples=30, *args, **kwargs)
        self.method='dbscan'

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
        nb of clusters.
        """
        self.mesh   = mesh
        self.kdtree = cKDTree(self.mesh)
        self.engine = None
        self.method='dummy'

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
        return self.kdtree.query(xs)[1]
        # return [self.kdtree.query(x)[1] for x in xs]


def make_clusterizer(xs, method='kmeans', **kwargs):
    """
    Clusterize the data with specified algorithm
    Naively assume you pass the right parameters for the right algo

    :param data:       array with shape (n,2) of inputs to clusterize
    :param method:     algo to use, supported: kmeans, dbscan, dummy
    :param n_clusters: nb of clusters to find (if applicable)

    :return: a clusterizer object (instance of child of Clusterizer())
    """
    
    clusterizer = None
    if method == 'kmeans':
        clusterizer = KmeansClusterizer(kwargs['nb_of_clusters'])
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

    # Obtain possible_outputs_list
    Z = clusterizer.predict(np.c_[xx.ravel(), yy.ravel()])

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
