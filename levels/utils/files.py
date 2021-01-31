from levels.tokens.mario import REPLACE_TOKENS
from levels.utils.conversion import ascii_to_one_hot_level


def opentxt(namefile):
    with open(namefile) as text:
        lines = text.readlines()
    return 0


def load_level_from_text(path_to_level_txt, replace_tokens=REPLACE_TOKENS):
    """ Loads an ascii level from a text file. """
    with open(path_to_level_txt, "r") as f:
        ascii_level = []
        for line in f:
            for token, replacement in replace_tokens.items():
                line = line.replace(token, replacement)
            ascii_level.append(line)
    return ascii_level


def read_level_from_file(file_path, replace_tokens=REPLACE_TOKENS):
    """ Returns a full token level tensor from a .txt file. Also returns the unique tokens found in this level.
    Token. """
    txt_level = load_level_from_text(file_path, replace_tokens)
    uniques = set()
    for line in txt_level:
        for token in line:
            # if token != "\n" and token != "M" and token != "F":
            if token != "\n" and token not in replace_tokens.items():
                uniques.add(token)
    uniques = list(uniques)
    uniques.sort()  # necessary! otherwise we won't know the token order later
    oh_level = ascii_to_one_hot_level(txt_level, uniques)
    return oh_level, uniques


def place_a_mario_token(level):
    """ Finds the first plausible spot to place Mario on. Especially important for levels with floating platforms.
    level is expected to be ascii."""
    # First check if default spot is available
    for j in range(1, 4):
        if level[-3][j] == '-' and level[-2][j] in ['X', '#', 'S', '%', 't', '?', '@', '!', 'C', 'D', 'U', 'L']:
            tmp_slice = list(level[-3])
            tmp_slice[j] = 'M'
            level[-3] = "".join(tmp_slice)
            return level

    # If not, check for first possible location from left
    for j in range(len(level[-1])):
        for i in range(1, len(level)):
            if level[i - 1][j] == '-' and level[i][j] in ['X', '#', 'S', '%', 't', '?', '@', '!', 'C', 'D', 'U', 'L']:
                tmp_slice = list(level[i - 1])
                tmp_slice[j] = 'M'
                level[i - 1] = "".join(tmp_slice)
                return level

    return level  # Will only be reached if there is no place to put Mario
