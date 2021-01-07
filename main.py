from config import cfg
from models.toadgan import TOADGAN
from levels.utils.files import read_level_from_text_file

if __name__ == "__main__":
    # Lock the settings
    cfg.freeze()

    # Load the level
    read_level_from_text_file()

    # Create and train the TOAD-GAN
    toad_gan = TOADGAN()
    toad_gan.train()
