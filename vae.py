import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

LAT_DIM = 2 * 2
OBS_DIM = 128
CAPACITY1 = 128 * 2 ** 5
CAPACITY2 = 64 * 2 ** 5
LRN_RATE = 1e-3
WEIGHT_DECAY = 1e-5
VAR_BETA = 1
VAE_EPOCH = 50
DROPOUT_PROB = 0.5

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.hidden1 = nn.Linear(
            in_features = OBS_DIM,
            out_features = CAPACITY1,
            bias = False,
        )

        self.hidden2 = nn.Linear(
            in_features = CAPACITY1,
            out_features = CAPACITY2,
            bias = False,
        )

        self.bn1 = nn.BatchNorm1d(CAPACITY1)
        self.bn2 = nn.BatchNorm1d(CAPACITY2)

        self.dropout1 = nn.Dropout(DROPOUT_PROB)
        self.dropout2 = nn.Dropout(DROPOUT_PROB)

        self.fc_mu = nn.Linear(
            in_features = CAPACITY2,
            out_features = LAT_DIM,
        )
        self.fc_logvar = nn.Linear(
            in_features = CAPACITY2,
            out_features = LAT_DIM,
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.hidden2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.hidden1 = nn.Linear(
            in_features = LAT_DIM,
            out_features = CAPACITY2,
            bias = False,
        )
        self.hidden2 = nn.Linear(
            in_features = CAPACITY2,
            out_features = CAPACITY1,
            bias = False,
        )

        self.dropout1 = nn.Dropout(DROPOUT_PROB)
        self.dropout2 = nn.Dropout(DROPOUT_PROB)

        self.bn1 = nn.BatchNorm1d(CAPACITY2)
        self.bn2 = nn.BatchNorm1d(CAPACITY1)

        self.output = nn.Linear(
            in_features = CAPACITY1,
            out_features = OBS_DIM,
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.hidden2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.output(x)
        x = x.view(x.size(0), OBS_DIM)
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

def reconstruction_error(recon_x, x):
    return F.mse_loss(
        recon_x.view(-1, OBS_DIM),
        x.view(-1, OBS_DIM),
        reduction = "sum",
    )

def vae_loss(recon_loss, mu, logvar):
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + VAR_BETA * kl_divergence

"""### Training and Evaluation


"""

def train_vae(
    vae, cnn, optimizer, device, train_dataloader, test_dataloader,
    evaluate = True,
):

    vae.train()

    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print('Number of parameters: %d' % num_params)

    if evaluate:
        train_loss, train_recon_loss = [], []
        test_train_loss, test_train_recon_loss = [], []
        test_eval_loss, test_eval_recon_loss = [], []

    print("Training: ", end = "")
    for epoch in range(VAE_EPOCH):
        if evaluate:
            train_loss.append(0)
            train_recon_loss.append(0)
            image_count = 0

        for image_batch, _ in train_dataloader:

            image_batch = image_batch.to(device)
            image_batch = cnn.penultimate_layers(image_batch)

            image_batch_recon, latent_mu, latent_logvar = vae(image_batch)
            #plot(image_batch, image_batch_recon, kind = "train")

            recon_loss = reconstruction_error(image_batch_recon, image_batch)
            loss = vae_loss(recon_loss, latent_mu, latent_logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if evaluate:
                train_loss[-1] += loss.item()
                train_recon_loss[-1] += recon_loss

                image_count += len(image_batch)

        if evaluate:
            vae.train()
            train_loss[-1] /= image_count
            train_recon_loss[-1] /= image_count

            vae.train()
            recon_loss_avg, loss_avg = eval_vae(vae, cnn, device, test_dataloader)
            test_train_loss.append(loss_avg)
            test_train_recon_loss.append(recon_loss_avg)

            vae.eval()
            recon_loss_avg, loss_avg = eval_vae(vae, cnn, device, test_dataloader)
            test_eval_loss.append(loss_avg)
            test_eval_recon_loss.append(recon_loss_avg)

            vae.train()

        print("%d, " % (epoch+1), end = "")

    print()
    if evaluate:
        return (
            vae, optimizer,
            [np.array(train_loss), np.array(train_recon_loss)],
            [np.array(test_train_loss), np.array(test_train_recon_loss)],
            [np.array(test_eval_loss), np.array(test_eval_recon_loss)],
        )
    return vae, optimizer

def eval_vae(vae, cnn, device, test_dataloader):

    test_loss, test_recon_loss = 0, 0
    image_count = 0
    for image_batch, _ in test_dataloader:

        with torch.no_grad():

            image_batch = image_batch.to(device)
            image_batch = cnn.penultimate_layers(image_batch)

            image_batch_recon, latent_mu, latent_logvar = vae(image_batch)
            #plot(image_batch, image_batch_recon, kind = "test")

            recon_loss = reconstruction_error(image_batch_recon, image_batch)
            loss = vae_loss(recon_loss, latent_mu, latent_logvar)

            test_recon_loss += recon_loss
            test_loss += loss.item()

            image_count += len(image_batch)

    return test_recon_loss / image_count, test_loss / image_count

def plot_loss(train_losses, test_train_losses, test_eval_losses, save_folder):
    plt.ion()

    plotlabels = ["Total error", "Reconstruction error", "KL divergence"]
    train_losses.append(train_losses[0] - train_losses[1])
    test_train_losses.append(test_train_losses[0] - test_train_losses[1])
    test_eval_losses.append(test_eval_losses[0] - test_eval_losses[1])

    ncols = 3
    fig, ax = plt.subplots(ncols = ncols, figsize = (9, 2.5))

    for i in range(ncols):
        ax[i].plot(train_losses[i], c = "blue", label = "training")
        ax[i].plot(test_train_losses[i], c = "green", label = "test (train)")
        ax[i].plot(test_eval_losses[i], c = "red", label = "test (eval)")

        ax[i].set_title(plotlabels[i])

    plt.tight_layout()
    plt.legend()
    #plt.show()
    plt.savefig(os.path.join(save_folder, "training_curve.png"), dpi = 600)

"""### Load VAE"""

def new_vae(
    device,
    lat_dim = 2 * 2,
    obs_dim = 128,
    capacity1 = 128 * 2 ** 5,
    capacity2 = 64 * 2 ** 5,
    lrn_rate = 1e-3,
    weight_decay = 1e-5,
    var_beta = 1,
    vae_epoch = 50,
):
    global LAT_DIM
    global OBS_DIM
    global CAPACITY1
    global CAPACITY2
    global LRN_RATE
    global WEIGHT_DECAY
    global VAR_BETA
    global VAE_EPOCH

    LAT_DIM = lat_dim
    OBS_DIM = obs_dim
    CAPACITY1 = capacity1
    CAPACITY2 = capacity2
    LRN_RATE = lrn_rate
    WEIGHT_DECAY = weight_decay
    VAR_BETA = var_beta
    VAE_EPOCH = vae_epoch

    vae = VariationalAutoencoder()
    vae = vae.to(device)

    optimizer = torch.optim.Adam(
        params = vae.parameters(),
        lr = LRN_RATE,
        weight_decay = WEIGHT_DECAY,
    )
    return vae, optimizer

def load_vae(folder_name, vae_name, device, optimizer_name = "adam.pth"):

    vae = VariationalAutoencoder()
    vae.load_state_dict(torch.load(os.path.join(folder_name, vae_name)))
    vae.to(device)
    optimizer = torch.optim.Adam(
        params = vae.parameters(),
        lr = LRN_RATE,
        weight_decay = WEIGHT_DECAY,
    )
    optimizer.load_state_dict(torch.load(
        os.path.join(folder_name, optimizer_name)
    ))
    return vae, optimizer
