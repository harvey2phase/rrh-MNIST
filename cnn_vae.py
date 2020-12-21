import os
import urllib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.utils

from cnn import ConvolutionalNeuralNet, create_and_train_cnn, freeze, load_cnn
from vae import new_vae, load_vae, train_vae
from load_mnist import load_mnist, to_numpy_arrays
from misc import plot, mkdir, make_exp_folder

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

# Train VAE -----------------------------------------------------------

exp_folder = make_exp_folder(RESULTS_FOLDER, "no_bn")

trained_epoch = 0
#vae, optimizer = load_vae("12-20_00-24", "vae_epoch=50.pth", device)
vae, optimizer = new_vae(device)

vae, optimizer, train_losses, test_train_losses, test_eval_losses = train_vae(
    vae, cnn, optimizer, device, train_dataloader, test_dataloader,
    evaluate = True,
)

torch.save(vae.state_dict(), os.path.join(
    exp_folder, "vae_epoch=" + str(trained_epoch + VAE_EPOCH) + ".pth"
))
torch.save(optimizer.state_dict(), os.path.join(exp_folder, "adam.pth"))
with open(os.path.join(exp_folder, "vae.txt"), "w+") as f:
    f.write(str(vae))
drive.flush_and_unmount()
drive.mount(GDRIVE)

plot_loss(train_losses, test_train_losses, test_eval_losses, exp_folder)

# Compute RRH ------------------------------------------------------------------

vae.train()
gammas, alphas, betas = calculate_rrh(vae, train_X, train_y)
plot_rrh(gammas, alphas, betas, filename = "het_train")

vae.eval()
gammas, alphas, betas = calculate_rrh(vae, test_X, test_y)
plot_rrh(gammas, alphas, betas, filename = "het_test")
