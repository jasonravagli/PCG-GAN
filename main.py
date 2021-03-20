import os
import shutil

from config import cfg
from files import level_files
from files.toadgan_project_files import save
from levels.toadgan_project import create_project, TOADGANProject
from ml.models.toadgan import TOADGAN
from ml.models.toadgan_single_scale import TOADGANSingleScale
from utils.converters import one_hot_to_ascii_level


def clean_training_dirs():
    if os.path.isdir(cfg.PATH.TRAIN.DIR):
        shutil.rmtree(cfg.PATH.TRAIN.DIR)
    os.mkdir(cfg.PATH.TRAIN.DIR)
    os.mkdir(cfg.PATH.TRAIN.SCALED_IMGS)
    os.mkdir(cfg.PATH.TRAIN.MONITOR_IMGS)
    os.mkdir(cfg.PATH.TRAIN.LOSSES)


if __name__ == "__main__":
    # Config from file and lock the settings
    cfg.merge_from_file("config.yaml")
    cfg.freeze()

    clean_training_dirs()

    # Load the level
    path_file_level = os.path.join(cfg.PATH.LEVELS_DIR, cfg.LEVEL.TYPE, cfg.LEVEL.NAME + ".json")
    level = level_files.load(path_file_level)

    # Create and train the TOAD-GAN
    toadgan = TOADGAN()
    toadgan.train(level, cfg.TRAIN.EPOCHS)

    # Create and save the trained TOAD-GAN as a project
    project = TOADGANProject()
    project.name = cfg.PATH.TRAIN.PROJECT_NAME
    project.toadgan = toadgan
    project.training_level = level

    print("\nSaving TOAD-GAN project...")
    save(cfg.PATH.TRAIN.DIR, project, save_training_info=True)
    print("Done")
