import torch
import numpy as np
import matplotlib.pyplot as plt

obj = torch.load("checkpoint_mnist.pt", map_location=torch.device('cpu'))

print(obj.keys())
# print(obj["epochs"])
# print(obj["timings"])
# print(obj["losses"])
n_epochs = obj["epochs"]
x = np.arange(n_epochs,step=1)
losses = np.array(obj["losses"])
plt.plot(x, losses[:,0], label="D")
plt.plot(x, losses[:,1], label="G")
plt.plot(x, losses[:,2], label="I")
plt.legend()
plt.show()

