import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from misc import mkdir

CNN_EPOCH = 14
LR = 1.0
GAMMA = 0.7
LOG_INT = 10


class ConvolutionalNeuralNet(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.penultimate_layers(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def penultimate_layers(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


def train_cnn(model, optimizer, device, dataloader):
    model.train()
    total_loss, samples = 0, 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        samples += len(data)

        """
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        """


def test_cnn(model, device, dataloader, train = False):
    model.eval()
    if train:
        model.train()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_len = len(dataloader.dataset)
    test_loss /= test_len

    if train == True:
        train = "train"
    else:
        train = "eval"
    print(
        'Test ({}) set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train, test_loss, correct, test_len, 100. * correct / test_len,
        )
    )


def create_and_train_cnn(
    device, train_dataloader, test_dataloader, save_folder,
):
    cnn = ConvolutionalNeuralNet().to(device)
    optimizer = torch.optim.Adadelta(cnn.parameters(), lr = LR)

    scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)
    for epoch in range(1, CNN_EPOCH + 1):
        train_cnn(cnn, optimizer, device, train_dataloader)
        #test_cnn(cnn, device, test_dataloader, train = True)
        test_cnn(cnn, device, test_dataloader)
        scheduler.step()

    date = datetime.date.today().strftime("%m-%d")
    model_name = date + "_epoch=" + str(CNN_EPOCH) + ".pth"
    optimizer_name = date + "_adadelta" + ".pth"
    mkdir(save_folder)
    torch.save(cnn.state_dict(), os.path.join(save_folder, model_name))
    torch.save(optimizer.state_dict(), os.path.join(save_folder, optimizer_name))
    return cnn


def load_cnn(model_name, device, load_folder, optimizer_name = None):
    cnn = ConvolutionalNeuralNet()
    cnn.load_state_dict(torch.load(os.path.join(load_folder, model_name)))
    cnn.to(device)
    return cnn


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
