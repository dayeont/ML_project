import torch
import os
import torch.nn as nn
import numpy as np
#from sklearn import datasets
from torch.utils.data import DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.utils.data as torch_data
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pickle
#import tqdm
from PIL import Image
import logging
import sys

logging.basicConfig(filename='pu_dist.out', level=logging.INFO)


torch.manual_seed(42)
mean1 = np.array([-2.5, 0])

cov1 = 0.5*np.eye(2)

gaus1 = np.random.multivariate_normal(mean1, cov1, 10000)
y1 = np.random.uniform(0, 3, 10000)
y2 = np.random.normal(2, 0.1, 10000)
gaus2 = np.array([np.sin(y1) + y2, 2*y1 -3]).T

gaus1 = torch.FloatTensor(gaus1)
gaus2 = torch.FloatTensor(gaus2)



lr = 10**(-3)
hidden_size = 30
latent_size = 2
batch_size = 100
beta1=0.5
n_epoch = int(10**5)
lambda_iter = 2
eta = 1e-3

class Gaussian(torch_data.Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]

gaus1_loader = DataLoader(Gaussian(gaus1), shuffle=True, batch_size=batch_size)
gaus2_loader = DataLoader(Gaussian(gaus2), shuffle=True, batch_size=batch_size)

# cost function 2-norm of given vectors difference
def c(x, y):
    return torch.norm(x - y)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
            
        self.net = nn.Sequential(
                nn.Linear(2, hidden_size),

                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),

                nn.LeakyReLU(),
                nn.Linear(hidden_size, 2),
            )
    def forward(self, x):
        output = self.net(x)
        return output
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
            
        self.net = nn.Sequential(
                nn.Linear(2, hidden_size),

                nn.LeakyReLU(),
                nn.Linear(hidden_size, 1)
            )
    def forward(self, x):
        output = self.net(x)
        output = output.mean(0)
        return output.view(1)

fixed_z = torch.FloatTensor(batch_size, 2).normal_(0, 1)
fake_x =[]
fake_y = []

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor if device == 'cuda:0' else torch.FloatTensor

# function for first weights normalization for generators and discriminators
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm1d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Initialization of models, weights of models, inputs
Gx = Generator()
Gy = Generator()

lambda_x = Discriminator()
lambda_y = Discriminator()

Gx.apply(weights_init)
Gy.apply(weights_init)

lambda_x.apply(weights_init)
lambda_y.apply(weights_init)

#Gx.to(device)
#Gy.to(device)
#lambda_x.to(device)
#lambda_y.to(device)

#x_cpu = Tensor(batch_size, 1, image_size, image_size).to(device)
#y_cpu = Tensor(batch_size, 3, image_size, image_size).to(device)
#z = Tensor(batch_size, latent_size, 1, 1).to(device)
#fixed_z = torch.FloatTensor(batch_size, latent_size, 1, 1).normal_(0, 1).to(device)
one = Tensor([1])#.to(device)
minus_one = Tensor([-1])#.to(device)
    
# Optimizers
parameters_lambda = list(lambda_x.parameters()) + list(lambda_y.parameters())
parameters_G = list(Gx.parameters()) + list(Gy.parameters())

optimizer_lambda = torch.optim.Adam(parameters_lambda, lr=lr, betas=(beta1, 0.999))
optimizer_G = torch.optim.Adam(parameters_G, lr=lr, betas=(beta1, 0.999))

# Training
for epoch in range(n_epoch):
    for i, (x, y) in enumerate(zip(gaus1_loader, gaus2_loader)):
        # if batch sizes are not equal continue
        if x.shape[0] != y.shape[0]:
            continue
            
        x_cpu = x
        y_cpu = y
        
        #x.to(device)
        #y.to(device)
        
        # update lambda parameters
        for param_x, param_y in zip(lambda_x.parameters(), lambda_y.parameters()):
            param_x.requires_grad = True
            param_y.requires_grad = True

        for j in range(lambda_iter):

            for param_x, param_y in zip(lambda_x.parameters(), lambda_y.parameters()):
                param_x.data.clamp_(-0.1, 0.1)
                param_y.data.clamp_(-0.1, 0.1)
                
            z = Tensor(batch_size, 2).normal_(0, 1).to(device)
            x = x.expand_as(y)

            lambda_x.zero_grad()
            lambda_y.zero_grad()

            real_lambda_x = lambda_x(x)
            real_lambda_y = lambda_y(y)
            
            real_lambda_x.backward(one)
            real_lambda_y.backward(one)

            fake_lambda_x = lambda_x(Gx(z).data)
            fake_lambda_y = lambda_y(Gy(z).data)
            
            fake_lambda_x.backward(minus_one)
            fake_lambda_y.backward(minus_one)
            err_lambda = real_lambda_x + real_lambda_y - fake_lambda_x - fake_lambda_y
            
            optimizer_lambda.step()
 
        # update generators parameters
        for param_x, param_y in zip(lambda_x.parameters(), lambda_y.parameters()):
            param_x.requires_grad = False
            param_y.requires_grad = False

        Gx.zero_grad()
        Gy.zero_grad()
        
        z = Tensor(batch_size, 2).normal_(0, 1)#.to(device)
        fake_x = Gx(z)
        fake_y = Gy(z)
        
        err_G = lambda_x(fake_x) + lambda_y(fake_y) + eta*c(fake_x, fake_y)
        err_G.backward(one)
        
        optimizer_G.step()

    if epoch % 1000 == 0:
        logging.info('Epoch {}/{} || Loss_lambda: {:.7f} || Loss_G {:.7f}'.format(epoch, n_epoch, err_lambda.data[0], err_G.data[0]))
        z = Tensor(1000, 2).normal_(0, 1).to(device)
        g1 = Gx(z).detach().numpy()
        g2 = Gy(z).detach().numpy()
        plt.figure(figsize=(10, 8))
        #plt.scatter(z[:, 0], z[:, 1], c=z[:, 1], cmap='spring')
        #plt.savefig('./1/gaus{}.png'.format(epoch))

        plt.scatter(g1[:, 0], g1[:, 1], c=g1[:, 1], cmap='winter')
        plt.scatter(g2[:, 0], g2[:, 1], c=g2[:, 1], cmap='autumn')
        plt.savefig('./2/2dist{}.png'.format(epoch))
