from turtle import forward
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

import sys

import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = np.loadtxt(open('mnist_train_small.csv','rb'),delimiter=',')

data = data[:,1:]

dataNorm = data / np.max(data)
dataNorm = 2*dataNorm - 1

dataT = torch.tensor(dataNorm).float()

batchsize = 100

class discriminatorNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256,256)
        self.out = nn.Linear(256,1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        
        x = self.fc2(x)
        x = F.leaky_relu(x)

        x = self.out(x)
        return torch.sigmoid(x)

class generatorNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(64,256)
        self.fc2 = nn.Linear(256,256)
        self.out = nn.Linear(256,784)

    def forward(self,x):
        x = self.fc1(x)
        x = F.leaky_relu(x)

        x = self.fc2(x)
        x = F.leaky_relu(x)

        x = self.out(x)
        return torch.tanh(x)



dnet = discriminatorNet().to(device)
gnet = generatorNet().to(device)

lossfun = nn.BCELoss()

d_optimizer = torch.optim.Adam(dnet.parameters(), lr=.0003)
g_optimizer = torch.optim.Adam(gnet.parameters(), lr=.0003)

num_epochs = 100000

losses = np.zeros((num_epochs,2))
discrim_decisions = np.zeros((num_epochs,2))

for epochi in range(num_epochs):
    
    randix = torch.randint(dataT.shape[0],(batchsize,))
    real_images = dataT[randix,:].to(device)
    fake_images = gnet(torch.randn(batchsize,64).to(device))

    real_labels = torch.ones(batchsize,1).to(device)
    fake_labels = torch.zeros(batchsize,1).to(device)

    # TRAIN THE DISCRIMINATOR

    # training on real pictures
    pred_real = dnet(real_images)
    d_loss_real = lossfun(pred_real,real_labels)

    # training of fake pictures
    pred_fake = dnet(fake_images)
    d_loss_fake = lossfun(pred_fake,fake_labels)

    # combined loss
    d_loss = d_loss_real + d_loss_fake
    losses[epochi,0] = d_loss.item()
    discrim_decisions[epochi,0] = torch.mean((pred_real>.5).float()).detach()

    # backprop
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # TRAIN THE GENERATOR
    
    # create fake images and compute loss
    fake_images = gnet(torch.randn(batchsize,64).to(device))
    pred_fake = dnet(fake_images)

    # compute and collect Loss and accuracy
    g_loss = lossfun(pred_fake,real_labels)
    losses[epochi,1] = g_loss.item()
    discrim_decisions[epochi,1] = torch.mean((pred_fake>.5).float()).detach()

    # backprop
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    if (epochi+1)%500==0:
        print(f"epoch: {epochi+1}")

fig,ax = plt.subplots(1,3,figsize=(18,5))

# generate the images from the generator network
gnet.eval()
fake_data = gnet(torch.randn(12,64).to(device)).cpu()

# and visualize...
fig,axs = plt.subplots(3,4,figsize=(8,6))
for i,ax in enumerate(axs.flatten()):
  ax.imshow(fake_data[i,:,].detach().view(28,28),cmap='gray')
  ax.axis('off')

plt.show()