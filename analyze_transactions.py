import metier

import numpy as np
import pandas as pd
import ipdb
from matplotlib import pyplot as plt
import seaborn as sns
import logging
from collections import Counter

from config.references import GRAPH_PATH
from config.manakin import (
        COLUMNS_TO_SCATTER,
        )

logging.basicConfig(level=logging.DEBUG)

class View_details():
    """
    This class is built with a raw_datas array containing transactions
    It receives update events with the indexes of selected transactions
    end then show them in a fancy way (actually scatterplot + boxplot of montants)
    """

    def __init__(self, raw_datas, columns_to_scatter=COLUMNS_TO_SCATTER, graph_path=GRAPH_PATH): #MODELE

        class Montant_plot(): #VUE

            def __init__(self, subplot):
                self.subplot = subplot

            def update(self, montants_dict):
                labels = []
                means = []
                stds = []
                logging.info('details_view: loading montants')
                for compte in montants_dict:
                    montants_mean = np.mean(montants_dict[compte])
                    means.append(float(montants_mean))
                    stds.append(float(np.std(montants_dict[compte])))
                    labels.append(compte)
                #ipdb.set_trace()
                ind = np.arange(len(means))
                logging.info('ind:'+str(ind)+' means:'+str(means)+' yerr:'+str(stds))
                self.subplot.clear()
                self.subplot.bar(ind, means, width=.7, yerr=stds)
                self.subplot.set_xticks(ind)
                self.subplot.set_xticklabels(labels)
                logging.info('details_view: montants ready')

        class Words_plot():
            def __init__(self, subplot):
                self.subplot = subplot
            def update(self):
                pass

        class Scatter_plot(): #VUE
            def __init__(self, graph_path):
                self.graph_path = graph_path

            def update(self, df):
                #self.subplot.pairplot(df)
                self.df = df
                g = sns.PairGrid(df, hue='account')
                g = g.map_diag(plt.hist)
                g = g.map_offdiag(plt.scatter)
                xlabels,ylabels = [],[]

                for ax in g.axes[-1,:]:
                    xlabel = ax.xaxis.get_label_text()
                    xlabels.append(xlabel)
                for ax in g.axes[:,0]:
                    ylabel = ax.yaxis.get_label_text()
                    ylabels.append(ylabel)

                for i in range(len(xlabels)):
                    for j in range(len(ylabels)):
                        g.axes[j,i].xaxis.set_label_text(xlabels[i])
                        g.axes[j,i].yaxis.set_label_text(ylabels[j])

                g.add_legend()
                g.savefig(self.graph_path + 'details.svg', format='svg', dpi=1200)
                g.savefig(self.graph_path + 'details.pdf', format='pdf', dpi=1200)
                logging.debug('transactions in cluster(s):\n'+str(df))

                return g.fig

        class Random_plot(): #VUE
            def __init__(self, subplot):
                self.subplot = subplot
            def update(self):
                self.subplot.plot(np.random.rand(50))
        
        #self.figure = figure
        self.graph_path = graph_path
        #self.montant_plot = Montant_plot(self.figure.add_subplot(2,2,1))
        self.scatter_plot = Scatter_plot(graph_path=self.graph_path)
        #self.details3_plot = Words_plot(self.figure.add_subplot(2,2,3))
        #self.details4_plot = Random_plot(self.figure.add_subplot(2,2,4))

        self.raw_datas = raw_datas
        self.columns_to_scatter = columns_to_scatter

    def update(self, idxs): #CONTROLEUR
        label_column = -4
        montant_column = 1
        max_to_display = 10

        logging.info("View_details: updating with "+str(len(idxs))+" accounts")

        labels = [ self.raw_datas[idx][label_column] for idx in idxs ]
        labels_count = Counter(labels)

        commons = np.array(labels_count.most_common())[:,0]
        logging.info("view_details: only display class "+str(commons))

        montant_dict = { label:[] for label in labels }
        transaction_list = []

        for i,idx in enumerate(idxs):
            current_x = self.raw_datas[idx]
            current_label = current_x[label_column] 
            transaction_list.append(current_x)

            if current_label in commons[:max_to_display]:
                montant_dict[current_label].append(current_x[montant_column])

        #self.montant_plot.update(montant_dict)

        df = metier.Annotations(transaction_list)
        for col in df.columns:
            if col not in self.columns_to_scatter:
                df = df.drop(col, axis=1)
            elif 'date' not in col:
                df[col] = df[col].convert_objects(convert_numeric=True)
        

        self.scatter_df   = df
        #ipdb.set_trace()
        self.montant_dict = montant_dict
        
        fig = self.scatter_plot.update(df.dropna())
        logging.info("View_details: ready")
        return fig


    def show(self):
        self.figure.show()
        #ipdb.set_trace()



