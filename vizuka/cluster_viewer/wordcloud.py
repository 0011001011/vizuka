from collections import Counter

import wordcloud
import matplotlib
matplotlib.use('Qt5Agg')  # noqa
from matplotlib import pyplot as plt

from vizuka.cluster_viewer.plotter import Plotter

class Wordcloud(Plotter):

    def __call__(self, data, fig, spec):
        """
        Finds occurrence of each word and draw a wordcloud

        :param: data is a list of the observations (list of sentences)
        :param: fig is the Figure on which the axe will be drawn
        :param:spec is the GridSpec for the axe

        :return: the axe with the visualization
        """
        if not data:
            return
        axe = plt.Subplot(fig, spec)

        data = [str(d) for d in data]
        words_freq = Counter(sum([phrase.split(' ') for phrase in data], []))
        del words_freq['']
        wc = wordcloud.WordCloud()
        
        wc.fit_words(words_freq)
        wc.background_color = 'white'
        wc.scale=10 # for better resolution
        wc.relative_scaling = .5 # for sizing words not only on ranks but also reative freq

        axe.axis('off')
        axe.imshow(wc.to_array())
        fig.add_subplot(axe)
        return axe
