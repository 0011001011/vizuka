import math

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

def cross_entropy(dict1, dict2):
    """
    Cross-entropy between two dicts
    dict1 must contains all keys in dict2
    """
    sum_dict1, sum_dict2 = sum(dict1.values()), sum(dict2.values())
    ce = 0

    for label in dict2:
        if dict1[label]==0 and dict2[label]==0:
            print("fuck that shit", label)
        elif dict1[label]==0:
            print("fuck that big shit", label)
        else:
            ce -= dict2[label]/sum_dict2*math.log(dict1[label]/float(sum_dict1))

    return ce

def bhattacharyya(dict1, dict2):
    """
    Similarity measure between two distribution
    
    :param dict1: dict with key:class, value:nb of observations
    """
    s = 0
    for i in {*dict1, *dict2}:
        s+=(dict1.get(i,0)*dict2.get(i,0))**.5
    return -math.log(s) if s!=0 else -np.inf

