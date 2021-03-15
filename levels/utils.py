import tensorflow as tf
import numpy as np


def downsample_image(image, target_shape, token_list, token_hierarchy):
    """
    Downsampling function customized to scale down level images. The code is adapted from the original
    TOAD-GAN project.

    image : Original level to be scaled down. Expects a numpy 3D tensor
    target_shape: The shape (height and width) to which the image will be scaled down
    token_list : List of ASCII tokens appearing in the image in order of channels from image.
    token_hierarchy: List of (group of) tokens that determines the tokens relevance (from the less to the more relevant).
    It is a list of dictionaries.
    """

    # Initial downscaling of one-hot level tensor is normal bilinear scaling
    bil_scaled = tf.image.resize(image, target_shape, method=tf.image.ResizeMethod.BILINEAR)

    # Init output level
    img_scaled = np.zeros(bil_scaled.shape, dtype=np.float32)

    for x in range(bil_scaled.shape[0]):
        for y in range(bil_scaled.shape[1]):
            curr_h = 0
            curr_tokens = [tok for tok in token_list if bil_scaled[x, y, token_list.index(tok)] > 0]
            for h in range(len(token_hierarchy)):  # find out which hierarchy group we're in
                for token in token_hierarchy[h].keys():
                    if token in curr_tokens:
                        curr_h = h

            for t in range(bil_scaled.shape[2]):
                if not (token_list[t] in token_hierarchy[curr_h].keys()):
                    # if this token is not on the correct hierarchy group, set to 0
                    img_scaled[x, y, t] = 0
                else:
                    # if it is, keep original value
                    img_scaled[x, y, t] = bil_scaled[x, y, t]

            # Adjust level to look more like the generator output through a Softmax function.
            img_scaled[x, y, :] = tf.nn.softmax(30*img_scaled[x, y, :]).numpy()

    return img_scaled
