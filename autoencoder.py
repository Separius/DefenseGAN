import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from utils import get_mnist_ds, mkdir
from modules import MLPAutoEncoder


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (x, _) in enumerate(train_loader):
        x = (x + 1) / 2
        x = x.to(device)
        optimizer.zero_grad()
        # output = model(torch.clamp(x + torch.randn_like(x) * 0.3, min=-1.0, max=1.0))
        output = model(x)
        loss = F.mse_loss(output, x)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = (data + 1) / 2
            data = data.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, data).item()
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    return test_loss


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=4, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(get_mnist_ds(32, True), batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(get_mnist_ds(32, False), batch_size=args.test_batch_size, shuffle=False, **kwargs)
    model = MLPAutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_test_loss = float('+inf')
    mkdir('./trained_models/')
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), './trained_models/mnist_ae_mlp.pt')
    print('best:', best_test_loss)
    model.load_state_dict(torch.load('./trained_models/mnist_ae_mlp.pt'))
    model.eval()
    for x, _ in train_loader:
        break
    x = (x + 1) / 2
    x = x[:64].to(device)
    with torch.no_grad():
        x_r = model(x)
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(x.cpu(), range=(0, 1.0), padding=5), (1, 2, 0)))
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Recon Images")
    plt.imshow(np.transpose(vutils.make_grid(x_r.cpu(), range=(0, 1.0), padding=5), (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    main()
