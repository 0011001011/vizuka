"""
Qt handler for drawing buttons and IHM black magic.
Please do read it you fool. I must admit I am not
proud of everything written down there.

This should be rewritten with QtCreator's help.
"""
import matplotlib
import sys
import logging
import os

matplotlib.use('Qt5Agg')  # noqa
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QDockWidget,
    QLineEdit,
    QInputDialog,
)

from vizuka.graphics import qt_helpers
from vizuka.cluster_diving import moar_filters
from vizuka import clustering
from vizuka import similarity


# logging.basicConfig(level=logging.DEBUG)
def onclick_wrapper(onclick):
    """
    This decorator for onclick detects if the mouse event
    should trigger something.
    I tried to make it worst but this is my best
    """
    def wrapper(*args, **kwargs):
        if args[0].detect_mouse_event:  # args[0] is self
            return onclick(*args, **kwargs)
        else:
            return lambda x: None
    return wrapper


class Qt_matplotlib_handler():

    def __init__(self, figure):
        self.right_dock = QtCore.Qt.RightDockWidgetArea
        self.textboxs = {}
        self.plottings = []

        # configure the app + main window
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.features_to_filter=None
        self.figure = figure


    def show(self):
        """
        Shows the window you built with your tears
        """
        self.window.showMaximized()
        for win in self.additional_windows:
            win.show()
        self.refresh()
        logging.info("refreshed")
        logging.info("showed")

    def refresh(self):
        """
        Refresh the matplotlib plots
        """
        for p in self.plottings:
            p.canvas.draw()
                
    
    @onclick_wrapper
    def onclick(self, *args, **kwargs):
        """
        ..seealso::onclick_wrapper
        """
        self.base_onclick(*args, **kwargs)

        
    def toogle_detect_mouse_event(self, *args, **kwargs):
        """
        Enable/Disable mouse_event handling ..seealso::onclick_wrapper
        """
        self.detect_mouse_event = not self.detect_mouse_event


class Viz_handler(Qt_matplotlib_handler):
    """
    A Viz_handler is attached to the viz_engine defined in vizualization.py
    It basically lists lots of QWidget things and propose methods to init
    them "gracefully", hum...

    Only IHM here.
    """
    
    def __init__(
            self,
            viz_engine,
            figure,
            onclick,
            additional_filters=[],
            additional_figures=[],
            ):
        """
        This object is a QtWindow (or 2-3-4...) with a mamtplotlib.Figure
        and a onclick event handler. As you can see it is also linked to
        a viz_engine which will do the magic.
        
        This instance should only do IHM stuff, nothing intellectual, everything
        is handled by the viz_engine
        """
        super(Viz_handler, self).__init__(figure)

        self.viz_engine = viz_engine
        self.detect_mouse_event = False
        self.base_onclick = onclick
        self.window.setWindowTitle('Data vizualization')
        self.additional_filters = additional_filters
        self.additional_figures = additional_figures

        self.additional_windows = []

        # add the main figure
        qt_helpers.add_figure(self.figure, window=self.window, plottings=self.plottings, onclick=self.onclick)
        #self.cluster_diver = Cluster_diver(self.x_raw, self.x_raw_columns, ['ape_code'])

        # add additional window
        if self.additional_filters:
            self.additional_windows.append(qt_helpers.add_window(self.window, 'moar filters'))
            moar_filters(
                    window=self.additional_windows[-1],
                    right_dock=QtCore.Qt.RightDockWidgetArea,
                    features = self.viz_engine.x_raw,
                    all_features_categories = self.viz_engine.x_raw_columns,
                    features_to_filter=additional_filters,
                    viz_engine=self.viz_engine,
                    )

        for fig in self.additional_figures:
            self.additional_windows.append(qt_helpers.add_window(self.window, 'Cluster viewer'))
            qt_helpers.add_figure(fig, self.additional_windows[-1], plottings=self.plottings)


        """
        if self.features_to_display:        
            self.cluster_window = add_window(self.window, 'Cluster view')
            add_cluster_view(
                    window=self.cluster_window,
                    right_dock = QtCore.Qt.RightDockWidgetArea,
                    features = self.viz_engine.x_raw,
                    all_features_categories = self.viz_engine.x_raw_columns,
                    features_to_diplay=features_to_display,
                    viz_engine=self.viz_engine,
        """


        # self.add_figure(self.additional_figure, onclick=None, window=self.additional_window)

        # logging.info("textboxs=adding")
        qt_helpers.add_checkboxes(
            self.window,
            "Filter by true class",
            self.viz_engine.possible_outputs_list,
            self.viz_engine.filter_by_correct_class,
            self.right_dock,
        )
        qt_helpers.add_checkboxes(
            self.window,
            "Filter by predicted class",
            self.viz_engine.possible_outputs_list,
            self.viz_engine.filter_by_predicted_class,
            self.right_dock,
        )
        qt_helpers.add_checkboxes(
            self.window,
            "Navigation options",
            ['detect mouse event'],
            self.toogle_detect_mouse_event,
            self.right_dock,
        )
        # logging.info("textboxs=ready")

        # add button
        # logging.info("action buttons=adding")
        #if self.viz_engine.x_raw:
        qt_helpers.add_button(
            self.window,
            "Export x",
            self.export,
            self.right_dock,
        )
        qt_helpers.add_button(
            self.window,
            "Save clusterization",
            self.save_clusters,
            self.right_dock,
        )

        # add menulist
        builtin_cl, extra_cl = clustering.list_clusterizer()
        qt_helpers.add_menulist(
            self.window,
            'Clustering method',
            'Clusterize', [*builtin_cl.keys(), *extra_cl.keys()],
            self.viz_engine.request_new_clustering,
            dockarea=self.right_dock,
        )

        builtin_fr, extra_fr = similarity.list_similarity()
        qt_helpers.add_menulist(
            self.window,
            'Clusters borders',
            'Delimits',
            [*builtin_fr.keys(), *extra_fr.keys()],
            self.viz_engine.request_new_frontiers,
            self.right_dock,
        )
        qt_helpers.add_menulist(
            self.window,
            'Predictor set',
            'Load',
            self.viz_engine.predictors,
            self.viz_engine.reload_predict,
            self.right_dock,
        )
        self.user_cluster_menulist = qt_helpers.add_menulist(
            self.window,
            'Saved clusters',
            'Load',
            self.viz_engine.saved_clusters,
            self.viz_engine.load_clusterization,
            self.right_dock,
        )
        self.textboxs['nb_of_clusters'] = qt_helpers.add_text_panel(
            self.window,
            'Number of clusters (default:120)',
            self.textbox_function_n_clusters,
            self.right_dock,
        )

    def save_clusters(self):
        text, validated = QInputDialog.getText(
                self.window,
                "YOLO", "Name for the clusters you selected ?"
                )
        if validated:
            if text == '':
                text = "clusters"
            self.viz_engine.save_clusterization(text+'.pkl')

    def textbox_function_n_clusters(self):
        """
        Wrapper for textbox, to change nb_of_clusters
        without specifying parameters
        """
        n_str = self.textboxs['nb_of_clusters'].text()
        n = int(n_str)
        self.viz_engine.nb_of_clusters = n
    
    def export(self):
        text, validated = QInputDialog.getText(
                self.window,
                "YOLO", "Name for the exported csv ?"
                )
        if validated:
            if text == '':
                text = "export.csv"
        self.viz_engine.export(text)
