import os
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QFileDialog

from config import cfg
from files import level_files, toadgan_project_files
from gui.model.toadgan_project import TOADGANProjectModel
from gui.view.main_window import MainWindow
from utils.converters import one_hot_to_ascii_level


class ControllerGeneration:
    def __init__(self, main_window: MainWindow, project_model: TOADGANProjectModel):
        self._main_window = main_window
        self._project_model = project_model

        # Connect controller to GUI signals
        main_window.connect_to_button_generate(self.generate_level)
        main_window.connect_to_button_load_project(self.load_project)
        main_window.connect_to_button_save_generated(self.save_generated_level)

    def generate_level(self):
        project = self._project_model.get_project()
        generated_oh_level = project.toadgan.generate_image()

        # Create a Level object for the generated level from the TOAD-GAN training level
        level = project.training_level.copy()
        level.level_oh = generated_oh_level
        level.level_size = generated_oh_level.shape[:2].as_list()
        level.level_ascii = one_hot_to_ascii_level(level.level_oh, level.unique_tokens)
        self._project_model.set_generated_level(level)

    def load_project(self):
        self._main_window.show_message_on_statusbar("Loading project...", fixed_message=True)

        # Show a dialog to select the level file
        start_directory = cfg.PATH.PROJECTS
        file_path = QFileDialog.getOpenFileName(self._main_window, "Load project file", directory=start_directory,
                                                filter="Project File (*.json)")[0]
        if file_path:
            project = toadgan_project_files.load(file_path)
            if project is not None:
                self._project_model.set_project(project)
                self._main_window.show_message_on_statusbar("Project Loaded")
            else:
                # Error loading project (invalid file)
                self._project_model.set_project(None)
                self._main_window.show_message_on_statusbar("Error Loading Project - Invalid File")
        else:
            self._main_window.show_message_on_statusbar("")

    def save_generated_level(self):
        # Show a dialog to select the destination file
        start_directory = os.path.join(cfg.PATH.LEVELS_DIR, "generated.json")
        file_path = QFileDialog.getSaveFileName(self._main_window, "Save level file", directory=start_directory,
                                                filter="Level File (*.json)")[0]
        if file_path:
            level_files.save(self._project_model.get_generated_level(), file_path)
            self._main_window.show_message_on_statusbar("Generated Level Saved")
