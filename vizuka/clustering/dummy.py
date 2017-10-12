from scipy.spatial import cKDTree

from vizuka.clustering.clustering import Clusterizer

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

