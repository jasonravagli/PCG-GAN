import json
import os
from tensorflow.keras.models import load_model

from files import level_files
from levels.toadgan_project import TOADGANProject
from ml.models.toadgan import TOADGAN
from ml.models.toadgan_single_scale import TOADGANSingleScale


def load(path_file: str) -> TOADGANProject:
    try:
        with open(path_file) as f:
            dict_project = json.load(f)

        project_dir = os.path.dirname(path_file)

        project = TOADGANProject()
        project.name = dict_project["name"]
        project.training_level = level_files.load(os.path.join(project_dir, dict_project["level"]))
        project.toadgan = TOADGAN()

        # Load and setup TOAD-GAN models
        conv_receptive_field = dict_project["conv-receptive-field"]
        scale_factor = dict_project["scale-factor"]
        list_scales = dict_project["scales"]

        project.toadgan.setup_network(project.training_level, conv_receptive_field, scale_factor)
        project.toadgan.list_gans = []
        for index_scale in range(project.toadgan.n_scales):
            scale_model = list_scales[index_scale]["model"]
            scale_noise_amplitude = list_scales[index_scale]["noise-amplitude"]

            # Reconstruction noise is not important when training is already done
            reconstruction_noise = None
            current_scale_gan = TOADGANSingleScale(img_shape=project.toadgan.scaled_images[index_scale].shape,
                                                   index_scale=index_scale,
                                                   get_generated_img_at_scale=project.toadgan.generate_img_at_scale,
                                                   get_reconstructed_img_at_scale=project.toadgan.get_reconstructed_image_at_scale,
                                                   reconstruction_noise=reconstruction_noise)
            # Load the generator model
            path_model = os.path.join(project_dir, dict_project["models-dir"], scale_model)
            current_scale_gan.init_generator_from_trained_model(path_model)
            # Set the noise amplitude used by the generator
            current_scale_gan.noise_amplitude = scale_noise_amplitude

            project.toadgan.list_gans.append(current_scale_gan)

        return project
    except:
        return None


def save(path_dir: str, project: TOADGANProject):
    project_path = os.path.join(path_dir, project.name)
    os.mkdir(project_path)

    dict_project = {
        "name": project.name,
        "level": "level.json",
        "tokenset": project.training_level.tokenset.name,
        "conv-receptive-field": project.toadgan.conv_receptive_field,
        "scale-factor": project.toadgan.scale_factor,
        "models-dir": "toadgan-scales"
    }

    # Save the level used for training inside the project directory
    level_files.save(project.training_level, os.path.join(project_path, dict_project["level"]))

    # Save the hierarchy of models composing the TOADGAN
    list_scales = []
    os.mkdir(os.path.join(project_path, dict_project["models-dir"]))
    for index_scale in range(project.toadgan.n_scales):
        model_name = f"scale-{index_scale}"
        dict_scale = {
            "model": model_name,
            "noise-amplitude": project.toadgan.list_gans[index_scale].noise_amplitude
        }

        scale_generator = project.toadgan.list_gans[index_scale].generator
        model_path = os.path.join(project_path, dict_project["models-dir"], model_name)
        scale_generator.save(model_path)

        list_scales.append(dict_scale)
    dict_project["scales"] = list_scales

    path_project_file = os.path.join(project_path, project.name + ".json")
    with open(path_project_file, "w") as f:
        json.dump(dict_project, f, indent=4)
