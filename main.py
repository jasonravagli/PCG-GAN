import os
import shutil

from config import cfg
from levels.tokens.mario import TOKEN_GROUPS
from models.toadgan import TOADGAN
from levels.utils.files import read_level_from_file, load_level_from_text
import levels


def get_tokens_data_structure_from_level_type(lvl_type):
    if lvl_type == "mario":
        return levels.tokens.mario.TOKEN_GROUPS, levels.tokens.mario.TOKEN_DOWNSAMPLING_HIERARCHY, levels.tokens.mario.REPLACE_TOKENS

    return None, None, None


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

    tk_groups, tk_hierarchy, tk_replace = get_tokens_data_structure_from_level_type(cfg.LEVEL.TYPE)

    # Load the level
    oh_level, tk_in_lvl = read_level_from_file(cfg.PATH.LEVEL, tk_replace)

    # Create and train the TOAD-GAN
    toad_gan = TOADGAN()
    toad_gan.train(oh_level, cfg.TRAIN.EPOCHS, tokens_in_lvl=tk_in_lvl, token_hierarchy=tk_hierarchy)
