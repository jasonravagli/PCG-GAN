from copied_code.models.discriminator import Discriminator
from copied_code.models.generator import Generator

def init_models(opt, input_shape):
    """ Initialize Generator and Discriminator. """
    # generator initialization:
    G = Generator(opt)
    # G.build([input_shape]*2)
    # G.summary()

    # discriminator initialization:
    D = Discriminator(opt)
    # D.build([input_shape]*2)
    # D.summary()

    return D, G