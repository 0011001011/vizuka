"""
Similarity calculator
This module is typically used to find the similartity between
different clusters.
"""
from vizuka.similarity import bhattacharyya
from vizuka.similarity import deterministic

def make_frontier(method):
    if method=='bhattacharyya':
        return bhattacharyya.compute_similarity
    elif method=='all':
        return deterministic.all_solid
    else:
        return deterministic.all_invisible

