import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # noqa
from matplotlib import pyplot as plt

from vizuka.cluster_viewer.plotter import Plotter

class Density(Plotter):

    @classmethod
    def get_help(self):
        h = "Use it to display the distribution of *numerical* values"
        return h

    def __call__(self, data, fig, spec):
        """
        Plot the density of your :param: data distribution

        :param: data is a list of the observations
        :param: fig is the Figure on which the axe will be drawn
        :param:spec is the GridSpec for the axe

        :return: the axe with the visualization
        """
        if not data:
            return
        axe = plt.Subplot(fig, spec)

        data = [float(d) for d in data]
        bins = 100 # int(len(data)/10)
        hist, bins = np.histogram(data, bins=bins)
        width = .7 *(bins[1] - bins[0])
        center = (bins[:-1] + bins[1:])/2
        
        axe.set_yscale('linear')
        axe.bar(center, hist, align='center', width=width)
        
        fig.add_subplot(axe)
        return axe

class LogDensity(Plotter):

    def __call__(self, data, fig, spec):
        """
        Plot the log-density of your :param: data distribution

        :param: data is a list of the observations
        :param: fig is the Figure on which the axe will be drawn
        :param:spec is the GridSpec for the axe

        :return: the axe with the visualization
        """
        if not data:
            return
        axe = plt.Subplot(fig, spec)

        data = [float(d) for d in data]
        bins = 100 # int(len(data)/10)
        hist, bins = np.histogram(data, bins=bins)
        width = .7 *(bins[1] - bins[0])
        center = (bins[:-1] + bins[1:])/2
        
        axe.set_yscale('log')
        axe.bar(center, hist, align='center', width=width)
        
        fig.add_subplot(axe)
        return axe

