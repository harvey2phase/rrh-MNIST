import os
import datetime
import urllib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.utils

from cnn import ConvolutionalNeuralNet, create_and_train_cnn, freeze
from load_mnist import load_mnist, to_numpy_arrays
from misc import plot, mkdir

GPU = True
device = torch.device("cuda:0" if GPU and torch.cuda.is_available() else "cpu")

sns.set()

MNIST_DATA = "/home/harveyw/scratch/data"
MY_DRIVE = "/home/harveyw/scratch"
RESULTS = os.path.join(MY_DRIVE, "rrh-MNIST-results/no_bn")

TEST = os.path.join(
    RESULTS, datetime.datetime.now().strftime("%m-%d_%H-%M"),
)
mkdir(TEST)


BATCH_SIZE = 64

train_dataloader = load_mnist(True, BATCH_SIZE, MNIST_DATA)
test_dataloader = load_mnist(False, BATCH_SIZE, MNIST_DATA)

train_X, train_y = to_numpy_arrays(train_dataloader)
test_X, test_y = to_numpy_arrays(test_dataloader)


"""## VAE"""

LAT_DIM = 2
OBS_DIM = 128
CAPACITY1 = 128 * 2 ** 5
CAPACITY2 = 64 * 2 ** 5
LRN_RATE = 1e-3
WEIGHT_DECAY = 1e-5
VAR_BETA = 1
VAE_EPOCH = 10
#DROPOUT_PROB = 0.5


"""# Experiments"""

cnn = create_and_train_cnn(device, train_dataloader, test_dataloader)
#cnn = load_cnn("12-19_epoch=14.pth")
freeze(cnn)
cnn.eval()


trained_epoch = 0

#vae, optimizer = load_vae("12-20_00-24", vae_name = "vae_epoch=50.pth")
vae, optimizer = new_vae()

vae, optimizer, train_losses, test_train_losses, test_eval_losses = train_vae(
    vae, optimizer, evaluate = True,
)

torch.save(vae.state_dict(), os.path.join(
    TEST, "vae_epoch=" + str(trained_epoch + VAE_EPOCH) + ".pth"
))
torch.save(optimizer.state_dict(), os.path.join(TEST, "adam.pth"))
with open(os.path.join(TEST, "vae.txt"), "w+") as f:
    f.write(str(vae))
drive.flush_and_unmount()
drive.mount(GDRIVE)

plot_loss(train_losses, test_train_losses, test_eval_losses)

vae.train()
gammas, alphas, betas = calculate_rrh(vae, train_X, train_y)
plot_rrh(gammas, alphas, betas, filename = "het_train")

vae.eval()
gammas, alphas, betas = calculate_rrh(vae, test_X, test_y)
plot_rrh(gammas, alphas, betas, filename = "het_test")
