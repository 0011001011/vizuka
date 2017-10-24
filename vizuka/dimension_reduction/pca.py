"""
Here is the code summoned to reduce the dimension of your
precious data, and also to load it.
We use t-SNE and, if you want, PCA just before it.

..note:: tSNE from sklearn is not the best but is standard
I suggest you to uncomment 'from MulticoreTSNE import TSNE as tsne'
as it will be much faster and won't crash if you use too much RAM.
However this needs extra-install steps :
-> cf https://github.com/DmitryUlyanov/Multicore-TSNE
"""

import logging

import numpy as np
from sklearn.decomposition import PCA as PCA_algorithm

from vizuka.dimension_reduction import projector


class PCA(projector.Projector):

    def __init__(self, nb_dimension=2, min_ratio_variance_explained=-1):
        """"
        Prepare a PCA projection, to a d-dimensional space or
        with to a space with the minimum number of dimension
        required to have a :param min_ratio_variance_explained:
        ratio of explained variance, if specified.

        :param nb_dimension: the number of dimension to keep, usually
        you want 2
        :param min_ratio_variance_explained: default to -1, if between
        [0,1] it will adjust the number of dimensions for the projection's
        explained variance ratio to match this param
        """
        self.min_ratio_variance_explained = min_ratio_variance_explained
        self.nb_dimension = nb_dimension
        self.parameters = [nb_dimension, min_ratio_variance_explained]
        
        self.engine = PCA_algorithm(svd_solver='randomized')


    def project(self, x):
        """
        Projects :param x:
        ..seealso: __init__
        """
        logging.info(   "starting PCA dimensional reduction,",
                        "needs an explained variance of {}\%".format(
                                self.min_ratio_variance_explained*100)
                        )

        self.engine.fit(x)

        if self.min_ratio_variance_explained:
            variance_explained = 0.
            nb_dim_to_keep     = 0

            while variance_explained < self.min_ratio_variance_explained:
                variance_explained += pca.explained_variance_ratio_[nb_dim_to_keep]
                nb_dim_to_keep += 1
        else:
            nb_dim_to_keep = self.nb_dimension

        x_pcaed          = pca.fit_transform(x)
        self.projections = x_pcaed[:,:nb_dim_to_keep]

        logging.info(   "PCA successfull,",
                        "{} dimensions (axis) where kept after orthnormalization".format(
                                nb_dim_to_keep)
                        )

        return self.projections
