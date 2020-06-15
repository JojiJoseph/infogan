import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(74, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1),
            nn.Linear(1024, 7*7*128),
            nn.BatchNorm1d(1)
        )
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.BatchNorm2d(1),
        )
    def forward(self, x):
        y = self.linear_layers(x)
        y = y.view(y.shape[0], 128, 7, 7)
        y = self.conv_layers(y)
        return y



class Discriminator(nn.Module):
    pass

if __name__ == "__main__":
    g = Generator()

    x = torch.ones(1, 1, 74)
    print(x.shape)
    y = g(x)

    import matplotlib.pyplot as plt
    plt.imshow(y[0].permute(1,2,0).squeeze().detach().numpy(), cmap="gray")
    plt.show()