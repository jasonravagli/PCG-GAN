from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QGridLayout, QFileDialog

from files import tokenset_files
from gui.model.level import LevelModel
from gui.model.tilebox import TileBoxModel
from gui.view.level_grid import LevelGrid
from gui.view.ui_main_window import Ui_MainWindow
from gui.view.widget_tilebox import WidgetTilebox


class MainWindow(QMainWindow):

    # Signal to notify the change of the grid size
    grid_size_changed = pyqtSignal(int, int)

    def __init__(self, tilebox_model: TileBoxModel, level_model: LevelModel):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Add the LevelGrid widget to the central panel
        self.level_grid = LevelGrid(level_model, tilebox_model)
        grid_layout = QGridLayout()
        grid_layout.addWidget(self.level_grid)
        self.ui.panel_grid.setLayout(grid_layout)

        # Add the WidgetTilebox to the left panel
        self.widget_tilebox = WidgetTilebox(tilebox_model)
        self.ui.panel_tilebox_scroll_area.layout().addWidget(self.widget_tilebox)
        # grid_layout = QGridLayout()
        # grid_layout.addWidget(self.widget_tilebox)
        # self.ui.panel_tilebox_.setLayout(grid_layout)

        self._tilebox_model = tilebox_model
        self._level_model = level_model

        # Register as observer of the LevelModel to update UI components on update
        level_model.observe(self.update_ui_on_level_changed)

        # Make the value changed of the spins to trigger the grid_size_changed event
        self.ui.spinBoxLevelRows.valueChanged.connect(lambda: self.emit_grid_size_changed())
        self.ui.spinBoxLevelColumns.valueChanged.connect(lambda: self.emit_grid_size_changed())

        # Load the available tilesets into the QComboBox
        self.ui.combo_tileset.insertItems(0, tokenset_files.get_all())

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

    def show_message_on_statusbar(self, message: str):
        self.ui.statusbar.showMessage(message, 2000)

    def update_ui_on_level_changed(self):
        self.blockSignals(True)
        rows, columns = self._level_model.get_level_size()
        self.ui.spinBoxLevelRows.setValue(rows)
        self.ui.spinBoxLevelColumns.setValue(columns)

        self.ui.edit_level_name.setText(self._level_model.get_name())
        self.blockSignals(False)

