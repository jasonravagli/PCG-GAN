from levels.level import Level
from ml.models.toadgan import TOADGAN


class TOADGANProject:
    def __init__(self):
        self.name = None
        self.toadgan = None
        self.training_level = None


def create_project(name: str, toadgan: TOADGAN, original_level: Level) -> TOADGANProject:
    project = TOADGANProject()
    project.name = name
    project.toadgan = toadgan
    project.training_level = original_level

    return project
