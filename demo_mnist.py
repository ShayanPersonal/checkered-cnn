import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from checkered_layers import CheckeredConv2d

class CheckeredCNN(nn.Module):
        # A tiny CCNN with 93,833 parameters. With minor data augmentations, achieves test errors competitive 
        # with Capsule Networks (8.2 million parameters)
        def __init__(self):
            super(CheckeredCNN, self).__init__()
            self.init_conv = CheckeredConv2d(1, 32, kernel_size=3, stride=1, padding=1)
            self.conv1 = CheckeredConv2d(32, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = CheckeredConv2d(32, 45, kernel_size=3, stride=2, padding=1)
            self.conv3 = CheckeredConv2d(45, 45, kernel_size=3, stride=1, padding=1)
            self.conv4 = CheckeredConv2d(45, 64, kernel_size=3, depth=2, stride=1, padding=1)
            self.bn1 = nn.BatchNorm3d(32)
            self.bn2 = nn.BatchNorm3d(32)
            self.bn3 = nn.BatchNorm3d(45)
            self.bn4 = nn.BatchNorm3d(45)
            self.bn5 = nn.BatchNorm3d(64)

            self.pool = nn.AdaptiveAvgPool3d(1)
            self.drop = nn.Dropout3d(0.2)
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            x = x.unsqueeze(2)
            x = F.relu(self.bn1(self.init_conv(x)))
            x = F.relu(self.bn2(self.conv1(x)))
            x = F.relu(self.bn3(self.conv2(x)))
            x = F.relu(self.bn4(self.conv3(x)))
            x = F.relu(self.bn5(self.conv4(x)))
            x = self.drop(x)
            vector = self.pool(x).view(x.size(0), -1)
            logits = self.fc(vector)
            return logits

def run(data_path, n_epochs=100):
    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            data.requires_grad_()
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 500 == 0:
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        acc = 100. * float(correct) / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(test_loader.dataset), acc))
        return acc

    model = CheckeredCNN().cuda()
    print("\nParameter count: {}\n".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    ))
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)

    train_dataset = datasets.MNIST(data_path, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomRotation(15),
                        transforms.RandomCrop(28, padding=2),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

    test_dataset = datasets.MNIST(data_path, train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

    best_acc = 0
    for epoch in range(1, n_epochs + 1):
        scheduler.step()
        train(epoch)
        with torch.no_grad():
            best_acc = max(test(), best_acc)
        print("Best test accuracy: {:.2f}%\n".format(best_acc))


if __name__ == "__main__":
    """
    Trains our example CCNN on MNIST, which achieves very high accuracy with just 93,833 parameters.
    Args:
        --data_path (string) - path to the directory with your MNIST dataset (will automatically download if it doesn't exist)

    To train the model:
    python demo_mnist.py --data_path <path_to_data_dir>

    Other args:
        --n_epochs (int) - number of epochs for training (default 100)
    """
    fire.Fire(run)