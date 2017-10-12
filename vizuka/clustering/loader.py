from vizuka.clustering import (
        kMeans,
        DBSCAN,
        dummy,
        )


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
        clusterizer = kMeans.KmeansClusterizer(kwargs['nb_of_clusters'])
    elif method == 'dbscan':
        clusterizer = DBSCAN.DBSCANClusterizer()
    else:
        clusterizer = dummy.DummyClusterizer(kwargs['resolution'])
    clusterizer.fit(xs)

    return clusterizer

