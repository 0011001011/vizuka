"""
Qt handler for drawing buttons and IHM black magic.
Please do read it you fool. I must admit I am not
proud of everything written down there.

This should be rewritten with QtCreator's help.
"""
import matplotlib
import sys
import logging

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
)

from vizuka.cluster_diving import moar_filters


# logging.basicConfig(level=logging.DEBUG)
def onclick_wrapper(onclick):
    """
    This decorator for onclick detects if the mouse event
    should trigger something.
    Hummmm...
    """
    def wrapper(*args, **kwargs):
        if args[0].detect_mouse_event:  # args[0] is self
            return onclick(*args, **kwargs)
        else:
            return lambda x: None
    return wrapper

def add_menulist(window, menu_name, button_name, categories, onlaunch, dockarea):
    """
    Add a menu list with action button

    :param menu_name:   displayed name of the list (displayed)
    :param button_name: the name of the button
    :param categories:  categories available for selection
    :param onlaunch:    action to trigger on click to button
    """

    root = window
    panel = QWidget()
    hbox = QHBoxLayout(panel)
    
    class MenuList(QtWidgets.QListWidget):

        def __init__(self, categories):
            QtWidgets.QListWidget.__init__(self)
            self.add_items(categories)
            self.itemClicked.connect(self.item_click)
            self.selected = categories

        def add_items(self, categories):
            for category in categories:
                item = QtWidgets.QListWidgetItem(category)
                self.addItem(item)

        def item_click(self, item):
            self.selected = str(item.text())
    
    menulist = MenuList(categories)

    hbox.addWidget(menulist)
    launchButton = QtWidgets.QPushButton(button_name)
    launchButton.clicked.connect(lambda: onlaunch(menulist.selected))
    hbox.addWidget(launchButton)
    panel.setLayout(hbox)
    
    dock = QDockWidget(menu_name, root)
    root.addDockWidget(dockarea, dock)
    dock.setWidget(panel)
    dock.resize(QtCore.QSize(dock.width(), dock.minimumHeight()))

    return menulist

def add_button(window, name, action, dockarea):
    """
    Adds a simple button

    :param name: diplayed button name
    :param action: function triggered on click event
    """
    root = window
    panel = QWidget()
    hbox = QHBoxLayout(panel)

    button = QtWidgets.QPushButton(name)
    button.clicked.connect(action)
    hbox.addWidget(button)
    panel.setLayout(hbox)

    dock = QDockWidget(name, root)
    root.addDockWidget(dockarea, dock)
    dock.setWidget(panel)


def add_figure(figure, window, plottings=[], onclick=None):
    """
    Easy method for adding a matplotlib figure to the Qt window

    :param figure:  matplotlib figure to attach
    :param onclick: onclick function to attach to figure
    :param window:  which window to attach the figure to
    """

    class MatplotlibWidget(QWidget):
        def __init__(self, figure, onclick, parent=None, *args, **kwargs):
            super(MatplotlibWidget, self).__init__(parent)
            self.figure = figure
            self.onclick = onclick
            
            class MplCanvas(FigureCanvas):

                def __init__(self, figure, onclick):
                    self.figure = figure
                    self.onclick = onclick
                    FigureCanvas.__init__(self, self.figure)
                    FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
                    FigureCanvas.updateGeometry(self)

                    # add mouse event
                    # logging.info("mouseEvents=adding")
                    if onclick is not None:
                        self.figure.canvas.mpl_connect('button_press_event', self.onclick)
                    # logging.info("mouseEvents=ready")
            
            self.canvas = MplCanvas(self.figure, self.onclick)
    
            self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
            self.canvas.setFocus()

            self.toolbar = NavigationToolbar(self.canvas, self)
            layout = QVBoxLayout()
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas)
            self.setLayout(layout)

    root = window
    panel = QWidget()
    plot_wrapper_box = QHBoxLayout(panel)

    plottings.append(
        MatplotlibWidget(figure=figure, onclick=onclick
                         ))
    plot_wrapper_box.addWidget(plottings[-1])

    panel.setLayout(plot_wrapper_box)
    dock = QDockWidget('', root)
    root.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
    dock.setWidget(panel)

def add_text_panel(window, name, update, dockarea):
    """
    Adds a text panel (how surprising) and binds it to a function

    :param name:i  diplayed name of Widget
    :param update: function to bind returnPressed event of textpanel
    """
    # ipdb.set_trace()
    root = window
    panel = QWidget()
    hbox = QHBoxLayout(panel)
    textbox = QLineEdit(parent=panel)

    textbox.returnPressed.connect(update)
    hbox.addWidget(textbox)
    panel.setLayout(hbox)

    dock = QDockWidget(name, root)
    root.addDockWidget(dockarea, dock)
    dock.setWidget(panel)

    return textbox

def add_window(parent_window, title):
    class Additional_Window(QMainWindow):
        def __init__(self, parent=None):
            super(Additional_Window, self).__init__(parent)
            self.setWindowTitle(title)
    return Additional_Window(parent_window)

def add_checkboxes(window, name, items_name, action, dockarea, checked_by_default=False):
    """
    Add some checkboxes, linked to some action.

    :param name:       the name of this checkboxes set
    :param items_name: displayed name of each item
    :param action:     triggered on each check/uncheck
    """
    root = window
    panel = QWidget()
    hbox = QHBoxLayout(panel)
    my_qlist = QtWidgets.QListWidget()
    my_item_list = {}

    default_state = QtCore.Qt.Checked if checked_by_default else QtCore.Qt.Unchecked

    for i in items_name:
        item = QtWidgets.QListWidgetItem()
        item.setText(str(i))
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(default_state)
        my_qlist.addItem(item)
        my_item_list[i] = item

    my_qlist.itemChanged.connect(
            lambda: action( {item_name: my_item_list[item_name].checkState() for item_name in items_name} ))
    hbox.addWidget(my_qlist)
    panel.setLayout(hbox)

    dock = QDockWidget(name, root)
    root.addDockWidget(dockarea, dock)
    dock.setWidget(panel)


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
        add_figure(self.figure, window=self.window, plottings=self.plottings, onclick=self.onclick)
        #self.cluster_diver = Cluster_diver(self.x_raw, self.x_raw_columns, ['ape_code'])

        # add additional window
        if self.additional_filters:
            self.additional_windows.append(add_window(self.window, 'moar filters'))
            moar_filters(
                    window=self.additional_windows[-1],
                    right_dock=QtCore.Qt.RightDockWidgetArea,
                    features = self.viz_engine.x_raw,
                    all_features_categories = self.viz_engine.x_raw_columns,
                    features_to_filter=additional_filters,
                    viz_engine=self.viz_engine,
                    )

        for fig in self.additional_figures:
            self.additional_windows.append(add_window(self.window, 'Cluster viewer'))
            add_figure(fig, self.additional_windows[-1], plottings=self.plottings)


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
        add_checkboxes(
            self.window,
            "Filter by true class",
            self.viz_engine.possible_outputs_list,
            self.viz_engine.filter_by_correct_class,
            self.right_dock,
        )
        add_checkboxes(
            self.window,
            "Filter by predicted class",
            self.viz_engine.possible_outputs_list,
            self.viz_engine.filter_by_predicted_class,
            self.right_dock,
        )
        add_checkboxes(
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
        add_button(
            self.window,
            "Export x",
            lambda: self.viz_engine.export(self.viz_engine.output_path),
            self.right_dock,
        )
        add_button(
            self.window,
            "Save clusterization",
            lambda: self.viz_engine.save_clusterization(),
            self.right_dock,
        )

        # add menulist
        add_menulist(
            self.window,
            'Clustering method',
            'Clusterize', ['KMeans', 'DBSCAN', 'Loader', 'Dummy'],
            self.viz_engine.request_new_clustering,
            dockarea=self.right_dock,
        )
        add_menulist(
            self.window,
            'Clusters borders',
            'Delimits',
            ['Bhattacharyya', 'All', 'None'],
            self.viz_engine.request_new_frontiers,
            self.right_dock,
        )
        add_menulist(
            self.window,
            'Predictor set',
            'Load',
            self.viz_engine.predictors,
            self.viz_engine.reload_predict,
            self.right_dock,
        )
        self.user_cluster_menulist = add_menulist(
            self.window,
            'Saved clusters',
            'Load',
            self.viz_engine.saved_clusters,
            self.viz_engine.load_clusterization,
            self.right_dock,
        )
        self.textboxs['nb_of_clusters'] = add_text_panel(
            self.window,
            'Number of clusters (default:120)',
            self.textbox_function_n_clusters,
            self.right_dock,
        )

    def textbox_function_n_clusters(self):
        """
        Wrapper for textbox, to change nb_of_clusters
        without specifying parameters
        """
        n_str = self.textboxs['nb_of_clusters'].text()
        n = int(n_str)
        self.viz_engine.nb_of_clusters = n

