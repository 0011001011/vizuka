#############################################
#
#  How to add a dimension reduction algorithm ?
#
##################
#
# Below is a working example. You should simply inherits
# vizuka.dimension_reduction.projector.Projector and implements
# a contructor and a project method.
#
# Typically in the __init__ you instanciate your engine (e.g the
# TSNE object from sklearn.manifold)  and /!\ register its parameters
# (see example below). This is important for vizuka to be able to
# ask the users the values of the parameters and not confuse different
# projections with different parameters.
#
# "project" method takes as input an array of your high-dimensional
# and returns an array with projection (/!\: set the self.projections
# object to your 2D embedding array (see below)).
#
######################################################################

#####################################################################
#
# from vizuka.dimension_reduction import projector
# from sklearn.manifold import TSNE
# 
# class tSNE(projector.Projector):
# 
#     def __init__(self, perplexity, learning_rate, n_iter):
#         
#         self.method         = 'tsne'
# 
#         self.register_parameters(
#                 parameters = {
#                     'perplexity'   : perplexity,
#                     'learning_rate': learning_rate,
#                     'n_iter'       : n_iter,
#                     }
#                 )
# 
#         self.engine = TSNE(
#             perplexity    = self.parameters['perplexity'],
#             learning_rate = self.parameters['learning_rate'],
#             n_iter        = self.parameters['n_iter'],
#             **multicore_parameters,
#             )
#         self.projections = []
# 
#     def project(self, x):
#         self.projections = self.engine.fit_transform(x)
#         return self.projections
