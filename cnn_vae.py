import os
import urllib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.utils

from cnn import ConvolutionalNeuralNet, create_and_train_cnn, freeze, load_cnn
from vae import new_vae, load_vae, train_vae, plot_loss
from load_mnist import load_mnist, to_numpy_arrays
from misc import plot, mkdir, make_exp_folder
from rrh import calculate_rrh, plot_rrh

GPU = True
device = torch.device("cuda:0" if GPU and torch.cuda.is_available() else "cpu")

sns.set()

# Static folder definitions ----------------------------------------------------

MNIST_DATA = "/home/harveyw/scratch/data"
MY_DRIVE = "/home/harveyw/scratch"
RESULTS_FOLDER = os.path.join(MY_DRIVE, "rrh-MNIST-results")
CNN_FOLDER = os.path.join(RESULTS_FOLDER, "cnn")

# Load MNIST -------------------------------------------------------------------

BATCH_SIZE = 64

train_dataloader = load_mnist(True, BATCH_SIZE, MNIST_DATA)
test_dataloader = load_mnist(False, BATCH_SIZE, MNIST_DATA)

train_X, train_y = to_numpy_arrays(train_dataloader)
test_X, test_y = to_numpy_arrays(test_dataloader)

# Train/load CNN ---------------------------------------------------------------

"""
cnn = create_and_train_cnn(
    device, train_dataloader, test_dataloader, CNN_FOLDER,
)
"""
cnn = load_cnn("12-21_epoch=14.pth", device, CNN_FOLDER)
freeze(cnn)
cnn.eval()

# Train VAE --------------------------------------------------------------------

def one_vae_experiment(
    lat_dim,
):
    exp_name = "dropout"
    exp_folder = make_exp_folder(RESULTS_FOLDER, exp_name)

    """
    vae, optimizer = load_vae(
        os.path.join(RESULTS_FOLDER, exp_name, "12-21_15-18"), "vae.pth", device,
    )
    """
    vae, optimizer = new_vae(
        device,
        lat_dim = lat_dim, lrn_rate = 1e-4, vae_epoch = 70,
    )

    vae, optimizer, train_losses, test_train_losses, test_eval_losses = train_vae(
        vae, cnn, optimizer, device, train_dataloader, test_dataloader,
        evaluate = True,
    )

    torch.save(vae.state_dict(), os.path.join(exp_folder, "vae.pth"))
    torch.save(optimizer.state_dict(), os.path.join(exp_folder, "adam.pth"))
    with open(os.path.join(exp_folder, "vae.txt"), "w+") as f:
        f.write(str(vae))

    plot_loss(train_losses, test_train_losses, test_eval_losses, exp_folder)

    # Compute RRH ------------------------------------------------------------------
    vae.train()
    gammas, alphas, betas = calculate_rrh(vae, cnn, device, train_X, train_y)
    plot_rrh(gammas, alphas, betas, exp_folder, "het_train")

    vae.eval()
    gammas, alphas, betas = calculate_rrh(vae, cnn, device, test_X, test_y)
    plot_rrh(gammas, alphas, betas, exp_folder, "het_test")

for lat_dim in range(1, 7):
    one_vae_experiment(lat_dim)
