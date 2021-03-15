import os
import shutil

from config import cfg
from files import level_files
from files.toadgan_files import save_project
from ml.models.toadgan import TOADGAN


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

    print("\nSaving TOAD-GAN project...")
    save_project(cfg.PATH.TRAIN.DIR, cfg.PATH.TRAIN.PROJECT_NAME, toadgan)
    print("Done")
