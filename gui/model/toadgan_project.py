from gui.model.observable import Observable
from levels.level import Level
from levels.toadgan_project import TOADGANProject


class TOADGANProjectModel(Observable):
    def __init__(self):
        super().__init__()

        self._project = None
        self._gen_level_size = None
        self._generated_level = None

    def get_generated_level(self) -> Level:
        return self._generated_level.copy() if self._generated_level is not None else None

    def get_gen_level_size(self) -> tuple:
        return self._gen_level_size

    def get_project(self) -> TOADGANProject:
        return self._project

    def set_generated_level(self, generated_level: Level):
        self._generated_level = generated_level
        self.notify()

    def set_gen_level_size(self, size: tuple):
        self._gen_level_size = size
        self.notify()

    def set_project(self, project: TOADGANProject):
        self._project = project
        if project is not None:
            self._gen_level_size = project.training_level.level_size
        self._generated_level = None
        self.notify()
