###########################################################################################
#
# How to add a new frontier engine (aka similarity measure) ?
#
##################################
#
# Simply creates a class in a new module that
# implements vizuka.similarity.similarity.Similarity
#
# Below is a simple example, to draw all frontiers (alpha is 1)
#
########################################################################



#######################################################
# WORKING EXAMPLE
##################
#
# from vizuka.similarity import similarity
#
# class AllFrontiers(similarity.Similarity):
#     def __init__(self):
#         pass
#     def compare(dict0, dict1, force_inside01=True):
#         """
#         Draw all frontiers
#         """
#         return 1 # Here 1 means "solid", 0 is "invisible. .5 for mid-solid
