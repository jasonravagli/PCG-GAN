import os

from config import cfg
from files.toadgan_project_files import load
from levels.level import Level
from utils.converters import one_hot_to_ascii_level

if __name__ == "__main__":
    # Config from file and lock the settings
    cfg.merge_from_file("config.yaml")
    cfg.freeze()

    print("\nLoading TOAD-GAN project...")
    project = load(os.path.join(cfg.PATH.TRAIN.DIR, cfg.PATH.TRAIN.PROJECT_NAME, cfg.PATH.TRAIN.PROJECT_NAME + ".json"))
    print("Done")

    folder_generation = os.path.join(cfg.PATH.TRAIN.DIR, "generation")
    if not os.path.isdir(folder_generation):
        os.mkdir(folder_generation)

    level = project.training_level.copy()
    for i in range(5):
        level.level_oh = project.toadgan.generate_image()
        level.level_size = level.level_oh.shape[:2]
        level.level_ascii = one_hot_to_ascii_level(level.level_oh, level.unique_tokens)
        image = level.render()
        image.save(os.path.join(folder_generation, f"generated-{i}.png"), format="png")
