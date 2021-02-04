import tensorflow as tf

from copied_code.level_utils import read_level
from levels.utils.level_image_gen import LevelImageGen
from copied_code.train import train
from levels.tokens.mario import REPLACE_TOKENS as MARIO_REPLACE_TOKENS
from copied_code.config import get_arguments, post_config


def get_tags(opt):
    """ Get Tags for logging from input name. Helpful for wandb. """
    return [opt.input_name.split(".")[0]]


def main():
    """ Main Training funtion. Parses inputs, inits logger, trains, and then generates some samples. """

    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # # Disable first GPU
    # tf.config.experimental.set_visible_devices(physical_devices[1:], 'GPU')

    # Parse arguments
    opt = get_arguments().parse_args()
    opt = post_config(opt)

    # Init game specific inputs
    replace_tokens = {}
    sprite_path = "resources/levels/" + opt.game + '/sprites'
    opt.ImgGen = LevelImageGen(sprite_path)
    replace_tokens = MARIO_REPLACE_TOKENS

    # Read level according to input arguments
    real = read_level(opt, None, replace_tokens)

    # Train!
    generators, noise_maps, reals, noise_amplitudes = train(real, opt)


if __name__ == "__main__":
    main()
