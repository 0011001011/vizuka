import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


class Clusterizer():

        def __init__(self, n_clusters, *args, **kwargs):
            """
            Builds a clusterizer object e.g: kmeans

            :param n_clusters: number of clusters
            :param *args **kwargs: parameters passed to the clusterizer engine
            """
            self.engine = None

        def predict(self, xs):
            """
            Gives the cluster in which the data is

            :params xys: array-like of (x,y) points
            :return: array-like of cluster id
            """
            return (0,)*len(xs)

class KmeansClusterizer(Clusterizer):

    def __init__(self, n_clusters, *args, **kwargs):
        self.engine = KMeans(n_clusters=n_clusters, *args, **kwargs)

    def fit(self, xs):
        self.engine.fit(xs)

    def predict(self, xs):
        return self.engine.predict(xs)

def clusterize(data, method='kmeans', n_clusters=120):
    """
    Clusterize the data with specified algorithm

    :param data: array with shape (n,2) of in put to clusterize
    :param method: algo to use, supported: kmeans
    :param n_clusters: number of clusters to find (if apply)
    """
    
    clusterizer = None

    if method == 'kmeans':
        clusterizer = KmeansClusterizer(n_clusters=n_clusters)
    clusterizer.fit(data)

    return clusterizer

def plot_clusters(data, clusterizer):
    """
    Nicely returns a viz of the clusters

    :param data: array with shape (n,2) of in put to clusterize
    :param clusterizer: ..seealso::clusterizer
    """

    h = .02     # point in the grid[x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels
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

    return f,ax

if __name__ == '__main__':
    import dim_reduction as dr
    datas_sets, models = dr.load_tSNE()
    datas = datas_sets[50, 1000, 'pca', 15000]

    clusterizer = clusterize(datas)
    f, ax = plot_clusters(datas, clusterizer)

    plt.show()
