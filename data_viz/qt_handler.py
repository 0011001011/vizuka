"""
Qt handler for drawing buttons and IHM black magic.
Please do read it you fool. I must admit I am not
proud of everything written down there.

This should be rewritten with QtCreator's help.
"""

import matplotlib
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

import sys
import logging


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


class Viz_handler():
    """
    A Viz_handler is attached to the viz_engine defined in vizualization.py
    It basically lists lots of QWidget things and propose methods to init
    them "gracefully", hum...

    Only IHM here.
    """

    def show(self):
        """
        Shows the window you built with your tears
        """
        logging.info("showing")
        self.window.showMaximized()
        # self.additional_window.show()
        self.refresh()
        sys.exit(self.app.exec_())

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

    def __init__(self, viz_engine, figure, onclick):
        """
        This object is a QtWindow (or 2-3-4...) with a mamtplotlib.Figure
        and a onclick event handler. As you can see it is also linked to
        a viz_engine which will do the magic.
        
        This instance should only do IHM stuff, nothing intellectual, everything
        is handled by the viz_engine
        """

        self.viz_engine = viz_engine
        self.figure = figure
        self.detect_mouse_event = False
        
        self.base_onclick = onclick
        self.plottings = []

        # configure the app + main window
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle('Data vizualization')

        # add the main figure
        self.add_figure(self.figure, self.onclick, window=self.window)

        # add additional window
        class Additional_Window(QMainWindow):
            def __init__(self, parent=None):
                super(Additional_Window, self).__init__(parent)
                self.setWindowTitle('Scatter Plot')
        # self.additional_window = Additional_Window(self.window)
        # self.add_figure(self.additional_figure, onclick=None, window=self.additional_window)

        right_dock = QtCore.Qt.RightDockWidgetArea

        # add textbox
        self.textboxs = {}
        # logging.info("textboxs=adding")
        self.add_checkboxes(
            "Filter by true class",
            self.viz_engine.possible_outputs_list,
            self.viz_engine.filter_by_correct_class,
            right_dock,
        )
        self.add_checkboxes(
            "Filter by predicted class",
            self.viz_engine.possible_outputs_list,
            self.viz_engine.filter_by_predicted_class,
            right_dock,
        )
        self.add_checkboxes(
            "Navigation options",
            ['detect mouse event'],
            self.toogle_detect_mouse_event,
            right_dock,
        )
        # logging.info("textboxs=ready")

        # add button
        # logging.info("action buttons=adding")
        self.add_button(
            "Export x",
            lambda: self.viz_engine.export(self.viz_engine.output_path),
            right_dock,
        )
        self.add_button(
            "View_details",
            lambda: self.viz_engine.view_details_figure(),
            right_dock,
        )
        # logging.info("action buttons=ready")

        # add menulist
        self.menulists = {}
        self.menulists['clustering_method'] = self.add_menulist(
            'Clustering method',
            'Clusterize', ['KMeans', 'DBSCAN', 'Dummy'],
            self.viz_engine.request_new_clustering,
            dockarea=right_dock,
        )
        self.menulists['clustering_method'] = self.add_menulist(
            'Clusters borders',
            'Delimits',
            ['Bhattacharyya', 'All', 'None'],
            self.viz_engine.request_new_frontiers,
            right_dock,
        )
        self.textboxs['number_of_clusters'] = self.add_text_panel(
            'Number of clusters (default:120)',
            self.textbox_function_n_clusters,
            right_dock,
        )
        self.menulists['predict_set'] = self.add_menulist(
            'Predictor set',
            'Load',
            self.viz_engine.predictors,
            self.viz_engine.reload_predict,
            right_dock,
        )

        # logging.info('Vizualization=ready')

    def add_figure(self, figure, onclick, window):
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

        self.plottings.append(
            MatplotlibWidget(figure=figure, onclick=onclick
                             ))
        plot_wrapper_box.addWidget(self.plottings[-1])

        panel.setLayout(plot_wrapper_box)
        dock = QDockWidget('', root)
        root.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        dock.setWidget(panel)
        
    def add_menulist(self, menu_name, button_name, categories, onlaunch, dockarea):
        """
        Add a menu list with action button

        :param menu_name:   displayed name of the list (displayed)
        :param button_name: the name of the button
        :param categories:  categories available for selection
        :param onlaunch:    action to trigger on click to button
        """

        root = self.window
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

    def add_button(self, name, action, dockarea):
        """
        Adds a simple button

        :param name: diplayed button name
        :param action: function triggered on click event
        """
        root = self.window
        panel = QWidget()
        hbox = QHBoxLayout(panel)

        button = QtWidgets.QPushButton(name)
        button.clicked.connect(action)
        hbox.addWidget(button)
        panel.setLayout(hbox)

        dock = QDockWidget(name, root)
        root.addDockWidget(dockarea, dock)
        dock.setWidget(panel)

    def add_text_panel(self, name, update, dockarea):
        """
        Adds a text panel (how surprising) and binds it to a function

        :param name:i  diplayed name of Widget
        :param update: function to bind returnPressed event of textpanel
        """
        # ipdb.set_trace()
        root = self.window
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

    def toogle_detect_mouse_event(self, *args, **kwargs):
        """
        Enable/Disable mouse_event handling ..seealso::onclick_wrapper
        """
        self.detect_mouse_event = not self.detect_mouse_event

    def add_checkboxes(self, name, items_name, action, dockarea):
        """
        Add some checkboxes, linked to some action.

        :param name:       the name of this checkboxes set
        :param items_name: displayed name of each item
        :param action:     triggered on each check/uncheck
        """
        root = self.window
        panel = QWidget()
        hbox = QHBoxLayout(panel)
        my_qlist = QtWidgets.QListWidget()
        my_item_list = {}

        for i in items_name:
            item = QtWidgets.QListWidgetItem()
            item.setText(str(i))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            my_qlist.addItem(item)
            my_item_list[i] = item

        my_qlist.itemChanged.connect(
                lambda: action( {item_name: my_item_list[item_name].checkState() for item_name in items_name} ))
        hbox.addWidget(my_qlist)
        panel.setLayout(hbox)

        dock = QDockWidget(name, root)
        root.addDockWidget(dockarea, dock)
        dock.setWidget(panel)

    def textbox_function_n_clusters(self):
        """
        Wrapper for textbox, to change number_of_clusters
        without specifying parameters
        """
        n_str = self.textboxs['number_of_clusters'].text()
        n = int(n_str)
        self.viz_engine.number_of_clusters = n

