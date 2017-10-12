from vizuka.heatmap import entropy
from vizuka.heatmap import accuracy

def make_heatmap(method):
    if method == 'entropy':
        return entropy.build
    elif method=='accuracy':
        return accuracy.build
