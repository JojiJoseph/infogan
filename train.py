import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from models.mnist_infogan import Generator, Discriminator

D = Discriminator()
G = Generator()

dname = os.path.dirname(__file__)

# Initial values of checkpoint data items
starting_epoch  = 0
losses = []
epoch_timings = []

if os.path.exists(os.path.join(dname, './checkpoint_mnist.pt')):
    checkpoint_mnist = torch.load(os.path.join(dname, './checkpoint_mnist.pt'), map_location=torch.device('cpu'))
    starting_epoch = checkpoint_mnist["epochs"]
    losses = checkpoint_mnist["losses"]
    epoch_timings = checkpoint_mnist["timings"]
    D.load_state_dict(checkpoint_mnist["D"])
    G.load_state_dict(checkpoint_mnist["G"])
    print("checkpoint is loaded")

D = D.cuda()
G = G.cuda()

criterion_ce = nn.CrossEntropyLoss()
criterion_bce = nn.BCELoss()
criterion_mse = nn.MSELoss()

datapath = "./data/" # Path to save models
eye10 = torch.eye(10).long().cuda()

dataset = MNIST(datapath,download=True,transform=Compose([Resize(32), ToTensor(), Normalize([0.5],[0.5])]))
testset = MNIST(datapath, train=False, transform=Compose([Resize(32), ToTensor(), Normalize([0.5],[0.5])]))

dataloader = DataLoader(dataset, 100, shuffle=True)
testloader = DataLoader(testset, 100, shuffle=True)

optim_D = Adam(D.parameters(), lr= 0.0002, betas=(0.5, 0.999))
optim_G = Adam(G.parameters(), lr= 0.0002, betas=(0.5, 0.999))
optim_GAN = Adam(itertools.chain(G.parameters(), D.parameters()), lr= 0.0002, betas=(0.5, 0.999))

def to_categorical(y, num_categories):
    y_cat = torch.sparse.torch.eye(num_categories).index_select(index=y, dim=0)
    return y_cat.float()

for epoch in range(starting_epoch,100):
    
    epoch_start_time = time.time()
    total_D_loss = 0.
    total_G_loss = 0.
    total_I_loss = 0.
    
    for id, (batch, label) in (enumerate(dataloader)):
        batch = batch.cuda()
        label = label.cuda()
        
        current_loss = {}

        # Train discriminator
        optim_D.zero_grad()
        optim_G.zero_grad()
        valid, y_pred, code = D(batch)
        loss = criterion_bce(valid.flatten(), torch.ones(batch.shape[0]).cuda())

        noise = torch.rand(batch.shape[0], 74).cuda()
        noise[:,62:72] = to_categorical(torch.randint(0, 10, size=(batch.shape[0],)), 10)
        noise[:,72:74] = torch.from_numpy(np.random.uniform(low=-1, high=1, size=(1,2))).cuda()
        fake_batch = G(noise)
        valid, y_pred, code = D(fake_batch)
        loss += criterion_bce(valid.flatten(), torch.zeros(batch.shape[0]).cuda())
        
        loss /= 2
        total_D_loss += loss.item()
        current_loss["D loss"] = loss.item()
        loss.backward()
        optim_D.step()

        # Train generator
        optim_G.zero_grad()
        optim_D.zero_grad()
        noise = torch.rand(batch.shape[0], 74).cuda()
        noise[:,62:72] = to_categorical(torch.randint(0, 10, size=(batch.shape[0],)), 10)
        noise[:,72:74] = torch.from_numpy(np.random.uniform(low=-1, high=1, size=(1,2))).cuda()
        fake_batch = G(noise)
        valid, y_pred, code = D(fake_batch)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        
        loss = criterion_bce(valid.flatten(), torch.ones(batch.shape[0]).cuda())
        loss.backward()
        current_loss["G loss"] = loss.item()
        total_G_loss += loss.item()
        optim_G.step()

        # Train for maximizing entropy
        optim_GAN.zero_grad()
        noise = torch.rand(batch.shape[0], 74).cuda()
        noise[:,72:74] = torch.from_numpy(np.random.uniform(low=-1, high=1, size=(batch.shape[0],2))).cuda()
        fake_batch = G(noise)
        valid, y_pred, code = D(fake_batch)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        loss = 0.2 * criterion_mse(code, noise[:,72:])
        loss +=  criterion_ce(y_pred, torch.argmax(noise[:,62:72], dim=1))
        current_loss["I loss"] = loss.item()
        total_I_loss += loss.item()

        loss.backward()
        optim_GAN.step()

        if id % 100 == 0 and id != 0:
            # current = losses[len(losses)-1]
            print("Epoch {}, batch: {} [{}] [{}] [{}]".format(epoch+1, id, current_loss["D loss"],current_loss["G loss"],current_loss["I loss"]))
            G.eval()
            img = []
            for i in range(10):
                row = []
                noise = torch.rand(1, 74).cuda()
                noise[:,62:72] = eye10[i]
                for c1 in np.linspace(-2, 2, 11):
                    noise[:,72:74] = torch.tensor([[c1, 0]])
                    fig = G(noise)
                    fig = fig.view(32,32).cpu().detach().numpy()*0.5 + 0.5
                    if len(row) == 0:
                        row = fig
                    else:
                        row = np.concatenate([row, fig], axis=1)
                if len(img) == 0:
                    img = row
                else:
                    img = np.concatenate([img, row])
            plt.imshow(img, cmap="gray")
            plt.savefig("c1/{}_{}".format(epoch, id))
            img = []
            for i in range(10):
                row = []
                noise = torch.rand(1, 74).cuda()
                noise[:,62:72] = eye10[i]
                for c2 in np.linspace(-2, 2, 11):
                    noise[:,72:74] = torch.tensor([[0, c2]])
                    fig = G(noise)
                    fig = fig.view(32,32).cpu().detach().numpy()*0.5 + 0.5
                    if len(row) == 0:
                        row = fig
                    else:
                        row = np.concatenate([row, fig], axis=1)
                if len(img) == 0:
                    img = row
                else:
                    img = np.concatenate([img, row])
            plt.imshow(img, cmap="gray")
            plt.savefig("c2/{}_{}".format(epoch, id))
            G.train()
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    epoch_timings.append(epoch_duration)
    losses.append([total_D_loss, total_G_loss, total_I_loss])
    checkpoint_mnist = {
        "epochs": epoch + 1,
        "losses": losses,
        "D": D.state_dict(),
        "G": G.state_dict(),
        "timings": epoch_timings 
    }
    torch.save(checkpoint_mnist, "checkpoint_mnist.pt")