import random

import matplotlib
matplotlib.use('Qt5Agg')  # noqa
from matplotlib import pyplot as plt
from matplotlib import gridspec

from vizuka.cluster_viewer.plotter import Plotter

class RandomImages(Plotter):

    def __call__(self, data, fig, spec):
        """
        Plots some random images found in the :param data: list given

        :param: data is a list of the observations
        :param: fig is the Figure on which the axe will be drawn
        :param:spec is the GridSpec for the axe

        :return: the axe with the visualization
        """
        if not data:
            return

        length = max(len(data)-1,0)
        selected_data = [data[random.randint(0,length)] for _ in range(25)]
        inner = gridspec.GridSpecFromSubplotSpec(
                        5,5,
                        subplot_spec=spec)
        for idx, inner_spec in enumerate(inner):
            axe = plt.Subplot(fig, inner_spec)
            axe.imshow(selected_data[idx])
            fig.add_subplot(axe)

        return axe
