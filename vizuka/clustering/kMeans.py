from sklearn.cluster import KMeans

from vizuka.clustering.clusterizer import Clusterizer


class KmeansClusterizer(Clusterizer):

    def __init__(self, number_of_clusters=15):
        """
        Uses sklearn kmeans
        """
        self.register_parameters(
                parameters={'number_of_clusters':number_of_clusters})

        self.method = 'kmeans'
        self.number_of_clusters = number_of_clusters

        self.engine = KMeans(
                n_clusters=int(float(self.number_of_clusters)),
                )

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
