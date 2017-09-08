"""
Similarity calculator
This module is typically used to find the similartity between
different clusters.
"""
import math
import numpy as np

def bhattacharyya(dict1, dict2, inside01=True):
    """
    Similarity measure between two empirical distribution
    
    :param dict1: dictionnary containing {class_label:occurence}
    :param dict2: dictionnary containing {class_label:occurence}
    :param inside01: value between 0 and 1, 1 being a complete
    similarity between the two clusters
    """
    s = 0
    max_value_possible = 0
    for i in {*dict1, *dict2}:
        s+=(dict1.get(i,0)*dict2.get(i,0))**.5
        if inside01:
            max_value_possible+= (max(dict1.get(i,0),dict2.get(i,0))**.5)
    if inside01:
        return -math.log(s)/max_value_possible if s!=0 else -np.inf
    else:
        return -math.log(s) if s!=0 else -np.inf


def all_are_similar(dict0, dict1, force_inside01=True):
    return 0

def all_are_dissimilar(dict0, dict1, force_inside01=True):
    return 1
