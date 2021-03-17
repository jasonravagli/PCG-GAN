import qdarkstyle as qdarkstyle
import sys

from PyQt5.QtWidgets import QApplication

from gui.controller.controller_generation import ControllerGeneration
from gui.controller.main_controller import MainController
from gui.model.level import LevelModel
from gui.model.tilebox import TileBoxModel
from gui.model.toadgan_project import TOADGANProjectModel
from gui.view.main_window import MainWindow

app = QApplication(sys.argv)
# Set the dark style for PyQt5
app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))

# Create the models
tilebox_model = TileBoxModel()
level_model = LevelModel()
toadgan_project_model = TOADGANProjectModel()

# Create the GUI
main_window = MainWindow(tilebox_model, level_model, toadgan_project_model)

# Create the controllers
main_controller = MainController(main_window, tilebox_model, level_model)
controller_generation = ControllerGeneration(main_window, toadgan_project_model)

main_window.show()
sys.exit(app.exec_())
