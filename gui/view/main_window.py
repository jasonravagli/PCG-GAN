from PIL.ImageQt import ImageQt
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QGridLayout

from files import tokenset_files
from gui.model.level import LevelModel
from gui.model.tilebox import TileBoxModel
from gui.model.toadgan_project import TOADGANProjectModel
from gui.view.designer_level_grid import DesignerLevelGrid
from gui.view.gen_level_image import GeneratedLevelImage
from gui.view.ui_main_window import Ui_MainWindow
from gui.view.widget_tilebox import WidgetTilebox


class MainWindow(QMainWindow):

    # Signal to notify the change of the grid size
    grid_size_changed = pyqtSignal(int, int)

    def __init__(self, tilebox_model: TileBoxModel, level_model_design: LevelModel, project_model: TOADGANProjectModel):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # ------ Level Design ------

        # Add the LevelGrid widget to the central panel
        self.level_grid = DesignerLevelGrid(level_model_design, tilebox_model)
        grid_layout = QGridLayout()
        grid_layout.addWidget(self.level_grid)
        self.ui.panel_grid.setLayout(grid_layout)

        # Add the WidgetTilebox to the left panel
        self.widget_tilebox = WidgetTilebox(tilebox_model)
        self.ui.panel_tilebox_scroll_area.layout().addWidget(self.widget_tilebox)

        self._tilebox_model = tilebox_model
        self._level_model = level_model_design

        # Register as observer of the LevelModel to update UI components on update
        level_model_design.observe(self.update_ui_on_level_changed)

        # Make the value changed of the spins to trigger the grid_size_changed event
        self.ui.spinBoxLevelRows.valueChanged.connect(lambda: self.emit_grid_size_changed())
        self.ui.spinBoxLevelColumns.valueChanged.connect(lambda: self.emit_grid_size_changed())

        # Load the available tilesets into the QComboBox
        self.ui.combo_tileset.insertItems(0, tokenset_files.get_all())

        # ------ Level Generation ------

        layout = QGridLayout()
        self.generated_level_image = GeneratedLevelImage()
        layout.addWidget(self.generated_level_image)
        layout.setSpacing(0)
        self.ui.panel_gen_level.setLayout(layout)

        self._project_model = project_model
        project_model.observe(self.update_ui_on_project_changed)
        self.update_ui_on_project_changed()

        self.ui.tabWidget.setCurrentIndex(0)

    # ---------------------- Level Design Methods ----------------------

    def connect_to_button_clear(self, slot):
        self.ui.button_clear.clicked.connect(slot)

    def connect_to_button_load_level(self, slot):
        self.ui.button_load_level.clicked.connect(slot)

    def connect_to_button_save(self, slot):
        self.ui.button_save.clicked.connect(slot)

    def connect_to_combo_tileset(self, slot):
        self.ui.combo_tileset.currentTextChanged.connect(slot)
        slot(self.ui.combo_tileset.currentText())

    def connect_to_edit_level_name(self, slot):
        self.ui.edit_level_name.textChanged.connect(slot)

    def connect_to_grid_size_changed(self, slot):
        self.grid_size_changed.connect(slot)

    def emit_grid_size_changed(self):
        rows = self.ui.spinBoxLevelRows.value()
        columns = self.ui.spinBoxLevelColumns.value()
        self.grid_size_changed.emit(rows, columns)

    def show_message_on_statusbar(self, message: str, fixed_message: bool = False):
        self.ui.statusbar.showMessage(message, 0 if fixed_message else 3000)

    def update_ui_on_level_changed(self):
        self.blockSignals(True)
        rows, columns = self._level_model.get_level_size()
        self.ui.spinBoxLevelRows.setValue(rows)
        self.ui.spinBoxLevelColumns.setValue(columns)

        self.ui.edit_level_name.setText(self._level_model.get_name())
        self.blockSignals(False)

    # ---------------------- Level Generation Methods ----------------------

    def connect_to_button_generate(self, slot):
        self.ui.button_generate.clicked.connect(slot)

    def connect_to_button_load_project(self, slot):
        self.ui.button_load_project.clicked.connect(slot)

    def connect_to_button_save_generated(self, slot):
        self.ui.button_save_generated.clicked.connect(slot)

    def update_ui_on_project_changed(self):
        project = self._project_model.get_project()

        # Project not loaded: reset and disable UI components
        if project is None:
            self.ui.edit_project_name.setText("")
            self.ui.label_orig_level_size.setText("Original Level Size - x -")
            self.ui.button_generate.setDisabled(True)
            self.ui.button_save_generated.setDisabled(True)
            self.generated_level_image.clear_pixmap()
        else:
            self.ui.edit_project_name.setText(project.name)
            self.ui.label_orig_level_size.setText(f"Original Level Size {project.training_level.level_size[0]} x "
                                                  f"{project.training_level.level_size[0]}")
            self.ui.button_generate.setDisabled(False)

            generated_level = self._project_model.get_generated_level()

            # No level has been generated: disable the save button and clear the image box
            if generated_level is None:
                self.ui.button_save_generated.setDisabled(True)
                self.generated_level_image.clear_pixmap()
            else:
                self.ui.button_save_generated.setDisabled(False)
                qt_image = ImageQt(generated_level.render())
                pixmap = QtGui.QPixmap.fromImage(qt_image)
                self.generated_level_image.set_level_pixmap(pixmap)
