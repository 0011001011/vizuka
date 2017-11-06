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
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt

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
    QMessageBox,
    QAction,
    QActionGroup,
    QMenu,
    QFileDialog,
    QDialog,
    QRadioButton,
    QLabel,
)

from vizuka.graphics import qt_helpers
from vizuka.cluster_diving import moar_filters
from vizuka import clustering
from vizuka import frontier
from vizuka.cluster_viewer import make_plotter, list_plotter


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

CLUSTER_PLOTTER = {}
            
class Cluster_viewer(matplotlib.figure.Figure):

    def __init__(self, features_to_display, x_raw, x_raw_columns, show_dichotomy=True):
        super().__init__()
        self.x_raw = x_raw
        self.x_raw_columns = x_raw_columns
        self.show_dichotomy = show_dichotomy # deprecated

        self.features_to_display = features_to_display
        self.spec_by_name = {}
        self.cluster_view_selected_indexes = []

        self.spec = gridspec.GridSpec(
                len(features_to_display.keys()),
                2, wspace=0.2
                )

        for idx,feature_name in enumerate(features_to_display.keys()):
            for plotter in features_to_display[feature_name]:
                self.spec_by_name[feature_name+plotter] = {}
                self.spec_by_name[feature_name+plotter]['good'] = self.spec[idx%2]
                self.spec_by_name[feature_name+plotter]['bad' ] = self.spec[idx%2+1]

                if plotter not in CLUSTER_PLOTTER.keys():
                    CLUSTER_PLOTTER[plotter] = make_plotter(plotter)
        
        def plot_help(plotter, plotter_name, data_name, fig, spec_to_update_, key, y):

            spec_to_update = spec_to_update_[key]
            axe = plt.Subplot(fig, spec_to_update)
            axe.axis("off")
            text = (
                    'Here will be displayed the distribution of the column {}\n'.format(data_name)
                    +"The visualisation tool '{}' will be used\n\n".format(plotter_name)
                    +'Click on a cluster to begin'
                    )
            axe.text(
                    0.5, 0.5,
                    text,
                    horizontalalignment='center',
                    verticalalignment=y,
                    )
            fig.add_subplot(axe)

            if 'log' in plotter_name:
                data_name += ' - log'
            data_name +=  ' - {} predictions'.format(key)
            if axe:
                axe.set_title(data_name)

        for key in ['good', 'bad']:
            for data_name in self.features_to_display:
                for plotter_name in self.features_to_display[data_name]:
                    plotter = CLUSTER_PLOTTER[plotter_name]
                    spec_to_update = self.spec_by_name[data_name+plotter_name]
                    y = 'top' if key=='good' else 'bottom'
                    plot_help(plotter, plotter_name, data_name, self, spec_to_update, key, y)

    def update_cluster_view(self, index_good, index_bad, data_by_index):
        """
        Updates the axes with the data of the selected cluster

        index_good: indexes of all good predictions
        index_bad: indexes of all bad predicitons
        data_by_index: data
        """
        self.clear()
        self.cluster_view_selected_indexes = set(index_good+index_bad)
        
        
        selected_xs_raw  ={}
        data_to_display = {}

        if self.show_dichotomy:
            selected_xs_raw['good'] = [
                    data_by_index[idx] for idx in self.cluster_view_selected_indexes if idx in index_good]
            selected_xs_raw['bad' ] = [
                    data_by_index[idx] for idx in self.cluster_view_selected_indexes if idx in index_bad ]
        else:
            selected_xs_raw['all'] = [
                    data_by_index[idx] for idx in self.cluster_view_selected_indexes]

        columns_to_display = [list(self.x_raw_columns).index(i) for i in self.features_to_display]
        if self.show_dichotomy:
            data_to_display['good'] = {
                self.x_raw_columns[i]:[x[i] for x in selected_xs_raw['good']]
                for i in columns_to_display
                }
            data_to_display['bad'] = {
                self.x_raw_columns[i]:[x[i] for x in selected_xs_raw['bad']]
                for i in columns_to_display
                }
        else:
            data_to_display['all'] = {
                    self.x_raw_columns[i]:[x[i] for x in selected_xs_raw['all']]
                    for i in columns_to_display
                    }

        def plot_it(plotter, plotter_name, data_to_display, data_name, fig, spec_to_update_, key):

            spec_to_update = spec_to_update_[key]
            data = data_to_display[key][data_name]
            axe = plotter(data, fig, spec_to_update)
            if 'log' in plotter_name:
                data_name += ' - log'
            data_name +=  ' - {} predictions'.format(key)
            if axe:
                axe.set_title(data_name)

        for key in ['good', 'bad']:
            for data_name in self.features_to_display:
                for plotter_name in self.features_to_display[data_name]:
                    plotter = CLUSTER_PLOTTER[plotter_name]
                    spec_to_update = self.spec_by_name[data_name+plotter_name]
                    plot_it(plotter, plotter_name, data_to_display, data_name, self, spec_to_update, key) 

        
    def reset(self):
        self.clear()
        self.cluster_view_selected_indexes = []


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
        for plotting in self.plottings:
            try:
                plotting.canvas.draw()
            except ValueError as e:
                # This exception is raised when nothing to be draw
                logging.warn(e)
                for i in self.figure.get_axes():
                    i.clear()
                
    
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
        self.detect_mouse_event = True
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
            self.add_figure(fig, 'Cluster viewer')

        # self.add_figure(self.additional_figure, onclick=None, window=self.additional_window)
        self.initUI()

    def add_figure(self, fig, name):
        self.additional_windows.append(qt_helpers.add_window(self.window, name))
        qt_helpers.add_figure(fig, self.additional_windows[-1], plottings=self.plottings)
        self.additional_windows[-1].show()

    
    def initUI(self):
        window = self.window
        
        window.statusBar().showMessage('Loading...')

        menubar  = window.menuBar()

        qt_helpers.add_simple_menu(
                "Load",
                {"Load a set of predictions":self.load_prediction},
                menubar,
                window,
                )
        qt_helpers.add_simple_menu(
                'Export',
                { 'Export data in selected clusters in csv':self.export },
                menubar,
                window,
                )

        clusteringMenu = menubar.addMenu("Clustering")

        saveclustersAct = QAction('Save current clusters', window)
        saveclustersAct.triggered.connect(self.save_clusters)
        clusteringMenu.addAction(saveclustersAct)

        requestClusteringMenu = QMenu('Request clustering', window)
        builtin_cl, extra_cl = clustering.list_clusterizer()
        available_clustering_engines = {**builtin_cl, **extra_cl}
        for clustering_engine_name in available_clustering_engines.keys():
            clusteringAct = QAction(clustering_engine_name, window)
            clusteringAct.triggered.connect(
                    lambda x, name=clustering_engine_name:self.request_new_clustering(name))
            requestClusteringMenu.addAction(clusteringAct)
        clusteringMenu.addMenu(requestClusteringMenu)

        self.loadClusterMenu = QMenu('Load saved clusters', window)
        for cluster_name in self.viz_engine.saved_clusters:
            loadAct = QAction(cluster_name, window)
            loadAct.triggered.connect(
                    lambda x, name=cluster_name:self.viz_engine.load_clusterization(name))
            self.loadClusterMenu.addAction(loadAct)
        clusteringMenu.addMenu(self.loadClusterMenu)
        
        frontierMenu = menubar.addMenu('Cluster frontiers')
        builtin_fr, extra_fr = frontier.list_frontiers()
        available_frontiers = {**builtin_fr, **extra_fr}
        for frontier_name in available_frontiers.keys():
            frontierAct = QAction(frontier_name, window)
            frontierAct.triggered.connect(
                    lambda x, name=frontier_name:self.viz_engine.request_new_frontiers(name))
            frontierMenu.addAction(frontierAct)
        
        filterMenu = menubar.addMenu('Filters')

        trueClassFilter       = QMenu('Filter by true class', window)
        self.all_true_filters = []

        for cls in self.viz_engine.possible_outputs_list:
            name = cls if len(cls)<30 else str(cls[:29])+'...'
            f = QAction(name, window, checkable=True)
            f.setChecked(True)
            f.class_ = cls
            self.all_true_filters.append(f)
            f.triggered.connect(lambda x:self.viz_engine.filter_by_correct_class(self.all_true_filters))
            trueClassFilter.addAction(f)
        
        def select_all_true(boolean=True):
            for f in self.all_true_filters:
                f.setChecked(boolean)
            self.viz_engine.filter_by_correct_class(self.all_true_filters)

        select_all = QAction("Select all", window)
        select_all.triggered.connect(lambda x:select_all_true(True))
        trueClassFilter.addAction(select_all)
        
        select_none = QAction("Unselect all", window)
        select_none.triggered.connect(lambda x:select_all_true(False))
        trueClassFilter.addAction(select_none)

        filterMenu.addMenu(trueClassFilter)
        
        predictClassFilter       = QMenu('Filter by predicted class', window)
        self.all_predicted_filters = []

        for cls in self.viz_engine.possible_outputs_list:
            name = cls if len(cls)<30 else str(cls[:29])+'...'
            f = QAction(name, window, checkable=True)
            f.setChecked(True)
            f.class_ = cls
            self.all_predicted_filters.append(f)
            f.triggered.connect(lambda x:self.viz_engine.filter_by_predicted_class(self.all_predicted_filters))
            predictClassFilter.addAction(f)

        def select_all_predicted(boolean=True):
            for f in self.all_predicted_filters:
                f.setChecked(boolean)
            self.viz_engine.filter_by_predicted_class(self.all_predicted_filters)

        select_all_p = QAction("Select all", window)
        select_all_p.triggered.connect(lambda x:select_all_predicted(True))
        predictClassFilter.addAction(select_all_p)
        
        select_none_p = QAction("Unselect all", window)
        select_none_p.triggered.connect(lambda x:select_all_predicted(False))
        predictClassFilter.addAction(select_none_p)

        filterMenu.addMenu(predictClassFilter)

        
        navigationMenu = menubar.addMenu('Navigation')
        mouseEnabledAction = QAction('Disable mouse click', window, checkable=True)
        mouseEnabledAction.triggered.connect(self.toogle_detect_mouse_event)
        navigationMenu.addAction(mouseEnabledAction)

        clusterviewMenu   = menubar.addMenu("Cluster exploration")
        clustersummaryAct = QAction("Add custom cluster info", window)
        clustersummaryAct.triggered.connect(self.request_cluster_viewer)
        clusterviewMenu.addAction(clustersummaryAct)

                
        colorMenu = menubar.addMenu("Color mode")
        color_modes = {
                'Colorize correctness of predictions':"by_prediction",
                'Colorize by classes':"by_true_class",
                }
        def on_color_radio_button_toggled(s):
            radiobutton = s.sender()
            if radiobutton.isChecked():
                self.viz_engine.request_color_mode(color_modes[radiobutton.column_name])
        radio_color_group = QActionGroup(colorMenu, exclusive=True)
        for i,mode in enumerate(color_modes.keys()):
            radiobutton = QAction(mode, radio_color_group, checkable=True)
            radiobutton.column_name = mode
            radiobutton.toggled.connect(lambda x:on_color_radio_button_toggled(radiobutton))
            if 'Colorize correctness of predictions' == mode:
                radiobutton.setChecked(True)
            colorMenu.addAction(radiobutton)
        

        
    def request_cluster_viewer(self):

        class AskPlotterWindow(QWidget):
            def __init__(self, column_name, viz_engine):
                QWidget.__init__(self)

                layout = QVBoxLayout()
                self.setLayout(layout)
                self.viz_engine = viz_engine
                self.column_name = column_name
                
                builtin_pl, extra_pl = list_plotter()
                available_pl = {**builtin_pl, **extra_pl}

                layout.addWidget(QLabel("The following visualization tools are available"),0)
                layout.addWidget(QLabel("Which one do you want to use for exploration ?"),1)

                for i,(plotter_name, plotter_class) in enumerate(available_pl.items()):
                    radiobutton = QRadioButton(plotter_name +' - '+plotter_class.get_help())
                    radiobutton.plotter_name = plotter_name
                    radiobutton.toggled.connect(self.on_radio_button_toggled)
                    layout.addWidget(radiobutton, i+2)

            def on_radio_button_toggled(self):
                radiobutton = self.sender()
                if radiobutton.isChecked():
                    self.viz_engine.add_cluster_view(self.column_name, [radiobutton.plotter_name])
                    self.destroy()


        class AskColumnWindow(QWidget):
            def __init__(self, viz_engine):
                QWidget.__init__(self)

                layout = QVBoxLayout()
                self.setLayout(layout)
                self.viz_engine = viz_engine

                layout.addWidget(QLabel("Your raw_data_"+self.viz_engine.version+".npz file has the following column\n"),0)
                layout.addWidget(QLabel("Which one do you want to explore ?"),1)

                for i,column in enumerate(viz_engine.x_raw_columns):
                    radiobutton = QRadioButton(column)
                    radiobutton.column_name = column
                    radiobutton.toggled.connect(self.on_radio_button_toggled)
                    layout.addWidget(radiobutton, i+2)

            def on_radio_button_toggled(self):
                radiobutton = self.sender()
                if radiobutton.isChecked():
                    self.kikoo = AskPlotterWindow(radiobutton.column_name, self.viz_engine)
                    self.kikoo.show()
                    self.destroy()
        
        self.kikoo = AskColumnWindow(self.viz_engine)
        self.kikoo.show()
        return self.kikoo

    def is_ready(self):
        self.window.statusBar().showMessage('Ready')

    def load_prediction(self):
        prediction_filename = self.file_dialog(
                "Load prediction",
                self.viz_engine.model_path,
                "{} Npz file (*{}.npz)".format(self.viz_engine.version, self.viz_engine.version),
                )
        if prediction_filename:
            self.viz_engine.reload_predict(prediction_filename)
        

    def is_loading(self):
        self.window.statusBar().showMessage('Loading...')

    def pop_dialog(self, msg):
        popup = QMessageBox(self.window)
        popup.setText(msg)
        popup.exec_()

    def file_dialog(self, name, directory, filter):
        dialog = QFileDialog(self.window, name, directory, filter)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.selectedFiles()[0]


    def save_clusters(self):
        text, validated = QInputDialog.getText(
                self.window,
                "YOLO", "Name for the clusters you selected ?"
                )
        if validated:
            if text == '':
                text = "clusters"
            cluster_name = self.viz_engine.save_clusterization(text)
            loadAct = QAction(cluster_name, self.window)
            loadAct.triggered.connect(lambda:self.viz_engine.load_clusterization(cluster_name))
            self.loadClusterMenu.addAction(loadAct)

    def export(self):
        text, validated = QInputDialog.getText(
                self.window,
                "YOLO", "Name for the exported csv ?"
                )
        if validated:
            if text == '':
                text = "export.csv"
        self.viz_engine.export(text)

    def request_new_clustering(self, clustering_engine_name):
        self.clustering_params={}
        for requested_param in clustering.get_required_arguments(clustering_engine_name):
            text, validated = QInputDialog.getText(
                    self.window,
                    "YOLO", requested_param+' ?'
                    )
            self.clustering_params[requested_param]=float(text)

        return self.viz_engine.request_new_clustering(clustering_engine_name, self.clustering_params)
