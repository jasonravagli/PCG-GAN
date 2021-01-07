def read_level_from_text_file(path_file, replace_tokens=None):
    """
    Returns a full token level tensor from a .txt file. Also returns the unique tokens found in this level.
    Token.
    """
    level_ascii = []
    uniques = set()
    with open(path_file, "r") as f:
        for line in f:
            # Eventually replace the undesired tokens
            if replace_tokens:
                for token, replacement in replace_tokens.items():
                    line = line.replace(token, replacement)
            level_ascii.append(line)

            # Search for uniques tokens in the level line
            for token in line:
                if token != "\n":
                    uniques.add(token)

    tokens_in_level = list(uniques)
    # Note: this autodetermines the one-hot encoding (it is not the user to specify it)
    tokens_in_level.sort()  # necessary! otherwise we won't know the token order later
    return level_ascii, tokens_in_level


def write_to_text_file():
    pass
