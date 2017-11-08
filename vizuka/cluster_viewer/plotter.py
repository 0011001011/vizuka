import matplotlib
matplotlib.use('Qt5Agg')  # noqa
from matplotlib import  pyplot as plt

class Plotter(object):
    """
    Callable function that returns an axe with a visualization
    Cf vizuka/.cluster_viewer.counter for a simple example
    """

    @classmethod
    def get_help(self):
        h = ""
        return h

    def __call__(self, data, fig, spec):
        """
        :param: data is a list of the observations
        :param: fig is the Figure on which the axe will be drawn
        :param:spec is the GridSpec for the axe

        :return: the axe with the visualization
        """
        axe = plt.Subplot(fig, spec)
        fig.add_subplot(axe)
        return axe
