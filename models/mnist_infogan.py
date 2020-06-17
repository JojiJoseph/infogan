import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(74, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 8*8*128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(8*8*128)
        )
        self.conv_layers = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )
    def forward(self, x):
        y = self.linear_layers(x)
        y = y.view(y.shape[0], 128, 8, 8)
        y = self.conv_layers(y)
        return y



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
        )
        self.fc = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU())
        self.d_layer = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid())
        self.q_discrete_layer = nn.Sequential(nn.Linear(1024, 10), nn.Softmax())
        self.q_continues_layer = nn.Sequential(nn.Linear(1024, 2), nn.LeakyReLU()) 
    def forward(self, x):
        y = self.conv_layers(x)
        y = y.view(x.shape[0], -1)
        y = self.fc(y)
        cat = self.q_discrete_layer(y)
        code = self.q_continues_layer(y)
        valid = self.d_layer(y)
        return valid, cat, code

if __name__ == "__main__":
    g = Generator()
    d = Discriminator()

    x = torch.ones(1, 1, 74)
    print(x.shape)
    y = g(x)
    out = d(y)
    print(out)
    import matplotlib.pyplot as plt
    plt.imshow(y[0].permute(1,2,0).squeeze().detach().numpy(), cmap="gray")
    plt.show()
