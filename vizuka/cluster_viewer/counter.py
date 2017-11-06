from collections import Counter as pyCounter

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # noqa
from matplotlib import pyplot as plt

from vizuka.cluster_viewer.plotter import Plotter

class Counter(Plotter):

    def __call__(self, data, fig, spec):
        """
        Show a counter of each class occurrences

        :param: data is a list of the observations
        :param: fig is the Figure on which the axe will be drawn
        :param:spec is the GridSpec for the axe
        :return: the axe with the visualization
        """
        if not data:
            return
        axe = plt.Subplot(fig, spec)
        c = pyCounter(data)
        x = [l for l in c.keys()]
        y = [c[l] for l in x]
        
        order = np.argsort(y)
        
        y = [y[i] for i in order]
        x = [x[i] for i in order]

        graduation = np.linspace(0, len(y), len(y))

        axe.bar(height=y, left=graduation)
        axe.set_xticks(graduation)
        axe.set_xticklabels([str(i) for i in x])
        fig.add_subplot(axe)

        return axe
