from config import cfg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.layers import ZeroPadding2D


# def pad_tensor(tensor, pad_size):
#     pad = ZeroPadding2D(pad_size)
#     return pad(tensor)


def plot_losses(path_file, c_wass_loss, c_gp_loss, g_adv_loss, g_rec_loss):
    iterations = range(len(c_wass_loss))

    fig = plt.figure()
    plt.plot(iterations, c_wass_loss, color="r", label="C Wass Loss")
    plt.plot(iterations, c_gp_loss, color="b", label="C GP Loss")
    plt.plot(iterations, g_adv_loss, color="g", label="G Adv Loss")
    plt.plot(iterations, g_rec_loss, color="m", label="G Rec Loss")
    plt.legend()
    fig.savefig(path_file)


def plot_lr(path_file, lr):
    iterations = range(len(lr))

    fig = plt.figure()
    plt.plot(iterations, lr, color="r", label="Learning Rate")
    plt.legend()
    fig.savefig(path_file)


def plot_noise_amplitude(path_file, noise_amplitude):
    iterations = range(len(noise_amplitude))

    fig = plt.figure()
    plt.plot(iterations, noise_amplitude, color="r", label="Noise Amplitude")
    plt.legend()
    fig.savefig(path_file)


def generate_noise(shape, std):
    # For some reason, implementation of SinGAN/TOAD-GAN is different from the paper
    # and a normal noise is generated and then multiplied by std
    return std*tf.random.normal(shape=shape)
