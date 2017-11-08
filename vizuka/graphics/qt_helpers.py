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
    QAction,
    QInputDialog,
)

def add_simple_menu(name, triggers_by_submenu_name, menubar, window):
    menu = menubar.addMenu(name)
    for subname, action in triggers_by_submenu_name.items():
        a = QAction(subname, window)
        a.triggered.connect(action)
        menu.addAction(a)



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
