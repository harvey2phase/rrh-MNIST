import os
import sys
import urllib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.utils

GPU = True
device = torch.device("cuda:0" if GPU and torch.cuda.is_available() else "cpu")

sns.set()

# Load local folders and modules -----------------------------------------------

DRIVE = "/home/harveyw/scratch"
MNIST_FOLDER = os.path.join(DRIVE, "data")
RRH_FOLDER = os.path.join(DRIVE, "rrh-MNIST")
curr_folder = os.getcwd()

sys.path.insert(1, os.path.join(curr_folder))
from feature_vae import new_vae, load_vae, train_vae, plot_loss

sys.path.insert(1, os.path.join(RRH_FOLDER, "cnn-vae"))
from cnn import ConvolutionalNeuralNet, create_and_train_cnn, freeze, load_cnn
from load_mnist import load_mnist, to_numpy_arrays
from rrh import calculate_rrh, plot_rrh, plot_rrh_matrices
from misc import plot, mkdir, make_exp_folder


# Load MNIST -------------------------------------------------------------------

BATCH_SIZE = 64

train_dataloader = load_mnist(True, BATCH_SIZE, MNIST_FOLDER)
test_dataloader = load_mnist(False, BATCH_SIZE, MNIST_FOLDER)

train_X, train_y = to_numpy_arrays(train_dataloader)
test_X, test_y = to_numpy_arrays(test_dataloader)

# Train/load CNN ---------------------------------------------------------------

CNN_FOLDER = os.path.join(RRH_FOLDER, "trained_cnn")
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
    exp_folder: str, lat_dim: int,
    train_gamma_matrix, train_alpha_matrix, train_beta_matrix,
    test_gamma_matrix, test_alpha_matrix, test_beta_matrix,
):

    vae, optimizer = new_vae(
        device,
        lat_dim = lat_dim, lrn_rate = 1e-4, vae_epoch = 70,
    )

    (
        vae, optimizer, train_losses, test_train_losses, test_eval_losses,
    ) = train_vae(
        vae, cnn, optimizer, device, train_dataloader, test_dataloader,
        evaluate = True,
    )

    torch.save(vae.state_dict(), os.path.join(exp_folder, "vae.pth"))
    torch.save(optimizer.state_dict(), os.path.join(exp_folder, "adam.pth"))
    with open(os.path.join(exp_folder, "vae.txt"), "w+") as f:
        f.write(str(vae))

    plot_loss(train_losses, test_train_losses, test_eval_losses, exp_folder)

    # Compute RRH --------------------------------------------------------------

    vae.eval()
    gammas, alphas, betas = calculate_rrh(vae, cnn, device, test_X, test_y)
    train_gamma_matrix.append(gammas)
    train_alpha_matrix.append(alphas)
    train_beta_matrix.append(betas)
    plot_rrh_matrices(
        train_gamma_matrix, train_alpha_matrix, train_beta_matrix,
        exp_folder, "het_test",
    )

    vae.train()
    gammas, alphas, betas = calculate_rrh(vae, cnn, device, train_X, train_y)
    test_gamma_matrix.append(gammas)
    test_alpha_matrix.append(alphas)
    test_beta_matrix.append(betas)
    plot_rrh_matrices(
        train_gamma_matrix, train_alpha_matrix, train_beta_matrix,
        exp_folder, "het_train",
    )


def experiments(N: int, lat_dim: int,  exp_folder: str):
    exp_folder = os.path.join(curr_folder, exp_folder)
    mkdir(exp_folder)
    train_gamma_matrix, train_alpha_matrix, train_beta_matrix = [], [], []
    test_gamma_matrix, test_alpha_matrix, test_beta_matrix = [], [], []

    for i in range(N):
        one_vae_experiment(
            exp_folder, lat_dim,
            train_gamma_matrix, train_alpha_matrix, train_beta_matrix,
            test_gamma_matrix, test_alpha_matrix, test_beta_matrix,
        )
N = 5
for lat_dim in range(3, 7):
    experiments(N, lat_dim, "lat_dim=" + str(lat_dim))
