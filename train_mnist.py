import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from models.mnist_infogan import Discriminator

net = Discriminator().cuda()
criterion_ce = nn.CrossEntropyLoss().cuda()
criterion_bce = nn.BCELoss().cuda()

datapath = "./data/"
eye10 = torch.eye(10).long().cuda()
dataset = MNIST(datapath,download=True,transform=ToTensor())
testset = MNIST(datapath, train=False, transform=ToTensor())
dataloader = DataLoader(dataset, 100, shuffle=True)
testloader = DataLoader(testset, 100, shuffle=True)

optim = Adam(net.parameters())
best_validation_accuracy = 0.
for epoch in range(5):
    for id, (batch, label) in tqdm(enumerate(dataloader)):
        optim.zero_grad()
        batch = batch.cuda()
        label = label.cuda()
        valid, y_pred, code = net(batch)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        loss = criterion_ce(y_pred, label)
        loss.backward()
        optim.step()
        if id > 100:
            break
    correct = 0.
    total = 0.
    for id, (batch, label) in (enumerate(testloader)):
        batch = batch.cuda()
        label = label.cuda()
        valid, y_pred, code = net(batch)
        total +=(batch.shape[0])
        correct += (torch.sum(torch.argmax(y_pred, dim=2).flatten() == label.flatten()).item())
        print(correct / total)
    if (correct/total > best_validation_accuracy):
        best_validation_accuracy = correct / total
        torch.save(net.state_dict(), "mnist.pt")
    