

class ConvolutionalNeuralNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

"""### Training and Evaluation"""

def train(model, optimizer):
    model.train()
    total_loss, samples = 0, 0

    for batch_idx, (data, target) in enumerate(train_dataloader):
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

def test(model, train = False):
    model.eval()
    if train:
        model.train()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= TEST_LEN

    if train == True:
        train = "train"
    else:
        train = "eval"
    print(
        'Test ({}) set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train, test_loss, correct, TEST_LEN, 100. * correct / TEST_LEN,
        )
    )

"""### Train and save CNN"""

def create_and_train_cnn():
    cnn = Net().to(device)
    optimizer = torch.optim.Adadelta(cnn.parameters(), lr = LR)

    scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)
    for epoch in range(1, CNN_EPOCH + 1):
        train(cnn, optimizer)
        test(cnn, train = True)
        test(cnn)
        scheduler.step()

    date = datetime.date.today().strftime("%m-%d")
    model_name = date + "_epoch=" + str(CNN_EPOCH) + ".pth"
    optimizer_name = date + "_adadelta" + ".pth"
    cnn_path = os.path.join(MY_DRIVE, "cnn")
    mkdir(cnn_path)
    torch.save(cnn.state_dict(), os.path.join(cnn_path, model_name))
    torch.save(optimizer.state_dict(), os.path.join(cnn_path, optimizer_name))
    return cnn

"""### Load saved CNN"""

def load_cnn(model_name, optimizer_name = None):
    cnn = Net()
    cnn.load_state_dict(torch.load(MY_DRIVE + "cnn/" + model_name))
    cnn.to(device)
    return cnn
