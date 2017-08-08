import matplotlib
matplotlib.use('Qt4Agg')  # noqa
from matplotlib.animation import TimedAnimation
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib import pyplot as plt
from PyQt4 import QtGui, QtCore

import sys
import logging
import ipdb

#logging.basicConfig(level=logging.DEBUG)


class Viz_handler():

    def show(self):
        self.window.show()
        self.plotting.canvas.draw()
        sys.exit(self.app.exec_())

    def refresh(self):
        self.plotting.canvas.draw()

    def __init__(self, viz_engine, figure, onclick):

        self.viz_engine = viz_engine
        self.figure = figure
        self.onclick = onclick
        
        # configure the app + main window
        self.app = QtGui.QApplication(sys.argv)
        self.window = QtGui.QMainWindow()
        self.window.setWindowTitle('Data vizualization')

        # add the main figure
        self.add_figure(self.figure, self.onclick)

        # add textbox
        self.textboxs = {}
        #logging.info("textboxs=adding")
        self.textboxs['show_only'] = self.add_text_panel(
            'Show one label',
            self.textbox_function_showonly
        )
        self.textboxs['show_all'] = self.add_text_panel(
            'Select all with label',
            self.textbox_function_showall
        )
        self.textboxs['n_clusters'] = self.add_text_panel(
                'Number of clusters (default:120)',
            self.textbox_function_n_clusters
        )
        #logging.info("textboxs=ready")

        # add button
        #logging.info("action buttons=adding")
        self.add_button("Export x", lambda :self.viz_engine.export(self.viz_engine.output_path))
        self.add_button("View_details", lambda :self.viz_engine.view_details_figure())
        #logging.info("action buttons=ready")

        # add menulist
        self.menulists = {}
        self.menulists['clustering_method'] = self.add_menulist(
                'Clustering method',
                'Clusterize', ['KMeans', 'Dummy'],
                self.viz_engine.request_new_clustering)
        self.menulists['clustering_method'] = self.add_menulist(
                'Borders',
                'Delimits',
                ['Bhattacharyya', 'All', 'None'],
                self.viz_engine.request_new_frontiers)

        #logging.info('Vizualization=ready')

    def add_figure(self, figure, onclick):

        class MatplotlibWidget(QtGui.QWidget):
            def __init__(self, figure, onclick, parent=None, *args, **kwargs):
                super(MatplotlibWidget, self).__init__(parent)
                self.figure = figure
                self.onclick = onclick
                
                class MplCanvas(FigureCanvas):

                    def __init__(self, figure, onclick):
                        self.figure = figure
                        self.onclick = onclick
                        FigureCanvas.__init__(self, self.figure)
                        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
                        FigureCanvas.updateGeometry(self)

                        # add mouse event
                        #logging.info("mouseEvents=adding")
                        self.figure.canvas.mpl_connect('button_press_event', self.onclick)
                        #logging.info("mouseEvents=ready")
                
                self.canvas = MplCanvas(self.figure, self.onclick)
        
                self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
                self.canvas.setFocus()

                self.toolbar = NavigationToolbar(self.canvas, self)
                layout = QtGui.QVBoxLayout()
                layout.addWidget(self.toolbar)
                layout.addWidget(self.canvas)
                self.setLayout(layout)

        root = self.window
        panel = QtGui.QWidget()
        plot_wrapper_box = QtGui.QHBoxLayout(panel)

        self.plotting = MatplotlibWidget(figure=figure, onclick=onclick)
        plot_wrapper_box.addWidget(self.plotting)

        panel.setLayout(plot_wrapper_box)
        dock = QtGui.QDockWidget('', root)
        root.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        dock.setWidget(panel)
        

    def add_menulist(self, menu_name, button_name, categories, onlaunch):
        """
        Add a menu list with action button

        :param menu_name: the name of the list (displayed)
        :param button_name: the name of the button
        :param categories: categories available for selection
        :param onlaunch: action to trigger on click to button
        """

        root = self.window
        panel = QtGui.QWidget()
        hbox = QtGui.QHBoxLayout(panel)
        
        class MenuList(QtGui.QListWidget):

            def __init__(self, categories):
                QtGui.QListWidget.__init__(self)
                self.add_items(categories)
                self.itemClicked.connect(self.item_click)
                self.selected = categories

            def add_items(self, categories):
                for category in categories:
                    item = QtGui.QListWidgetItem(category)
                    self.addItem(item)

            def item_click(self, item):
                self.selected = str(item.text())
        
        menulist = MenuList(categories)

        hbox.addWidget(menulist)
        launchButton = QtGui.QPushButton(button_name)
        launchButton.clicked.connect(lambda: onlaunch(menulist.selected))
        hbox.addWidget(launchButton)
        panel.setLayout(hbox)
        
        dock = QtGui.QDockWidget(menu_name, root)
        root.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        dock.setWidget(panel)

        return menulist

    def add_button(self, name, action):
        """
        Adds a simple button
        """
        root = self.window
        panel = QtGui.QWidget()
        hbox = QtGui.QHBoxLayout(panel)

        button = QtGui.QPushButton(name)
        button.clicked.connect(action)
        hbox.addWidget(button)
        panel.setLayout(hbox)

        dock = QtGui.QDockWidget(name, root)
        root.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        dock.setWidget(panel)

    def add_text_panel(self, name, update):
        """
        Adds a text panel (how surprising) and binds it to a function

        :param name: name of Widget
        :param update: function to bind returnPressed event of textpanel
        """

        #ipdb.set_trace()

        root = self.window
        panel = QtGui.QWidget()
        hbox = QtGui.QHBoxLayout(panel)
        textbox = QtGui.QLineEdit(parent=panel)

        textbox.returnPressed.connect(update)
        hbox.addWidget(textbox)
        panel.setLayout(hbox)

        dock = QtGui.QDockWidget(name, root)
        root.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        dock.setWidget(panel)

        return textbox

    def textbox_function_showonly(self):
        """
        Wrapper for textbox, to use self.update_showonly
        without specifying parameters
        """
        class_str = self.textboxs['show_only'].text()
        if class_str == '':
            self.viz_engine.reset_viz()
        else:
            class_ = int(class_str)
            self.viz_engine.update_showonly(class_)


    def textbox_function_showall(self):
        """
        Wrapper for textbox, to use self.update_showall
        without specifying parameters
        """
        class_str = self.textboxs['show_all'].text()
        if class_str == '':
            self.viz_engine.reset_viz()
        else:
            class_ = int(class_str)
            self.viz_engine.update_showall(class_)
    
    def textbox_function_n_clusters(self):
        """
        Wrapper for textbox, to change n_clusters
        without specifying parameters
        """
        n_str = self.textboxs['n_clusters'].text()
        n = int(n_str)
        self.viz_engine.n_clusters = n

