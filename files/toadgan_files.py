import json

import os

from files import level_files
from ml.models.toadgan import TOADGAN


def save_project(path_dir: str, project_name: str, toadgan: TOADGAN):
    project_path = os.path.join(path_dir, project_name)
    os.mkdir(project_path)

    dict_project = {
        "level": "level.json",
        "tokenset": toadgan.training_level.tokenset.name,
        "conv-receptive-field": toadgan.conv_receptive_field,
        "scale-factor": toadgan.scale_factor,
        "models-dir": "toadgan-scales",
        "models-scales": [f"scale-{scale}" for scale in range(toadgan.n_scales)]
    }

    # Save the level used for training inside the project directory
    level_files.save(toadgan.training_level, os.path.join(project_path, dict_project["level"]))

    # Save the hierarchy of models composing the TOADGAN
    os.mkdir(os.path.join(project_path, dict_project["models-dir"]))
    for scale in range(toadgan.n_scales):
        scale_generator = toadgan.list_gans[scale].generator
        model_path = os.path.join(project_path, dict_project["models-dir"], dict_project["models-scales"][scale])
        scale_generator.save(model_path)

    path_project_file = os.path.join(project_path, project_name + ".json")
    with open(path_project_file, "w") as f:
        json.dump(dict_project, f, indent=4)
