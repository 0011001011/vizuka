import math
import numpy as np
"""
Cool functions for ML
"""

def entropy(my_dic):
    """
    StraightForward entropy calculation

    :param my_dict: dict of occurence of different classes
    :return: discrete entropy calculation
    """
    effectif_total = sum(my_dic.values())
    s = 0
    for effectif in my_dic.values():
        proportion = effectif / effectif_total
        if proportion > 0:
            s += proportion * math.log(proportion)
    return -s

def cross_entropy(global_distribution, specialized_distribution, global_entropy=None):
    """
    Cross-entropy between two dicts
    dict1 must contains all keys in specialized_distribution

    :param gobal_distribution: dictionnary containing {class_label:occurence}
    :param specialized_distribution: dictionnary containing {class_label:occurence}
    """

    specialized_array = np.array(specialized_distribution)

    if global_entropy is None:
        global_array = np.array(global_distribution)
        global_entropy = np.log(global_array / np.sum(global_array))
        if len(global_array) == 0:  # case where there's nothing
            return 0

    entropy_local = specialized_array / np.sum(specialized_array)
    
    return np.sum( - entropy_local * global_entropy)
    

    # return ce

def bhattacharyya(dict1, dict2):
    """
    Similarity measure between two empirical distribution
    
    :param dict1: dictionnary containing {class_label:occurence}
    :param dict2: dictionnary containing {class_label:occurence}
    """
    s = 0
    for i in {*dict1, *dict2}:
        s+=(dict1.get(i,0)*dict2.get(i,0))**.5
    return -math.log(s) if s!=0 else -np.inf

