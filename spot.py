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
from PIL import Image
from utils import Logger
import sys

sys.stdout = Logger('./pu_gaus.out', 'w', sys.stdout)

lr = 10**(-4)
eta = 10**(-3)
clamp_down = -0.01
clamp_up = 0.01
beta1 = 0.5
batch_size = 32
batch_size_test = 32
image_size = 32 
latent_size = 100 # Z
hidden_g = 4   # number of hidden layers for generators
hidden_d = 5   # number of hidden layers for discriminators
n_epoch = int(2*10**5)
lambda_iter = 10

transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), 
                                transforms.Normalize([0.5], [0.5])])
                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# MNIST(https://pytorch.org/docs/stable/torchvision/datasets.html )
# tranforming PIL Images to tensor
mnist_train = Datasets.MNIST('./Datasets/', download=False, 
                                     train=True,
                                     transform = transform
                            )

mnist_test = Datasets.MNIST('./Datasets/', download=False, 
                                     train=False,
                                     transform = transform
                           )

# loading mnist train and test data
# (torch.Size([bath_size, 1, image_size, image_size]), torch.Size([batch_size])
mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size,
                                          shuffle=True)
mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size_test,
                                         shuffle=True)


#MNISTM

# load MNIST-M images from pkl file
with open('./Datasets/mnistm_data.pkl', "rb") as f:
    mnist_m_data = pickle.load(f, encoding='bytes')
    
mnist_m_data_train = torch.cat([transform(Image.fromarray(mnist_m_data['train']['data'][i], mode='RGB')).reshape(1, 3, image_size, image_size)
                               for i in range(len(mnist_m_data['train']['data']))], 0)

mnist_m_data_test = torch.cat([transform(Image.fromarray(mnist_m_data['test']['data'][i], mode='RGB')).reshape(1, 3, image_size, image_size)  
                               for i in range(len(mnist_m_data['test']['data']))], 0)

mnist_m_label_train = torch.tensor(mnist_m_data['train']['labels'])
mnist_m_label_test = torch.tensor(mnist_m_data['test']['labels'])


class MNISTM(torch_data.Dataset):
    def __init__(self, X, y):
        super(MNISTM, self).__init__()
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dset = MNISTM(mnist_m_data_train, mnist_m_label_train) 
test_dset = MNISTM(mnist_m_data_test, mnist_m_label_test) 

# (torch.Size([bath_size, 3, image_size, image_size]), torch.Size([batch_size])
mnistm_train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
mnistm_test_loader = DataLoader(test_dset, batch_size=batch_size_test, shuffle=False) 

# form Appendix.A.1(No-sharing Network)
class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        assert latent_size == 100, 'length of Z should be 100'
        
        # for generators should be 4 hidden convolutional layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=latent_size, 
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=512, 
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=256, 
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=128, 
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64, 
                out_channels=3,
                kernel_size=2,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()
        )
    #INPUT: Z in R^100,  N(0, I)
    def forward(self, z):
        # if isinstance(z.data, torch.cuda.FloatTensor)
        # output = nn.parallel.data_parallel(self.main, z, range(self.ngpu))
        output = self.deconv(z)
        return output
     
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3, 
                out_channels=64,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True), # in the paper ReLU
            nn.Conv2d(
                in_channels=64, 
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=128, 
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=256, 
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=512, 
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False
            )
        )
    #INPUT: image x in R^(64x64x3) in mu or nu
    def forward(self, x):
        output = self.conv(x)
        output = output.mean(0)
        #print(output.shape)#?
        return output.view(1) #?

# cost function for MNIST and MNISTM (adding knowledge about countour)

def c(x, y, channels=3):
    C1 = Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) 
    C2 = Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    conv1 = nn.Conv2d(
        in_channels = channels, 
        out_channels = 1, 
        kernel_size=3, 
        bias=False)
    
    conv2 = nn.Conv2d(
        in_channels = channels, 
        out_channels = 1, 
        kernel_size=3, 
        bias=False)
    
    conv1.weight.data[0, :, :, :] = C1
    conv2.weight.data[0, :, :, :] = C2
    
    conv1.weight.data.requires_grad = False
    conv2.weight.data.requires_grad = False
    
    return torch.norm(conv1(x) - conv1(y)) + torch.norm(conv2(x) - conv2(y))


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
Tensor = torch.FloatTensor #torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

# function for first weights normalization for generators and discriminators
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Initialization of models, weights of models, inputs
Gx = Generator(latent_size)
Gy = Generator(latent_size)

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
fixed_z = Tensor(batch_size, latent_size, 1, 1).normal_(0, 1)#.to(device)
one = Tensor([1])#.to(device)
minus_one = Tensor([-1])#.to(device)
    
# Optimizers
parameters_lambda = list(lambda_x.parameters()) + list(lambda_y.parameters())
parameters_G = list(Gx.parameters()) + list(Gy.parameters())

optimizer_lambda = torch.optim.Adam(parameters_lambda, lr=lr, betas=(beta1, 0.999))
optimizer_G = torch.optim.Adam(parameters_G, lr=lr, betas=(beta1, 0.999))


for epoch in range(n_epoch):
    data_iter_x = iter(mnist_test_loader)
    data_iter_y = iter(mnistm_test_loader)
    i = 0
    # for i, ((x, _), (y, _)) in enumerate(zip(mnist_test_loader, mnistm_test_loader)):
    while i < len(mnist_test_loader) - 1:#5001:
        # if batch sizes are not equal continue
        #if x.shape[0] != y.shape[0]:
        #    continue
        #x = Tensor(x)
       # y = Tensor(y)

       # x_cpu = x
       # y_cpu = y

        # update lambda parameters
        for param_x, param_y in zip(lambda_x.parameters(), lambda_y.parameters()):
            param_x.requires_grad = True
            param_y.requires_grad = True
        #lambda_x.weight.requires_grad = True
        #lambda_x.bias.requires_grad = True

        #lambda_y.weight.requires_grad = True
        #lambda_y.bias.requires_grad = True
        j = 0
        #for j in range(lambda_iter):
        while j < 3 and i < len(mnist_test_loader) - 1: #5001:

                j += 1
                i += 1

                x, _ = data_iter_x.next()
                y, _ = data_iter_y.next()

                x_cpu = x
                y_cpu = y
                if x.shape[0] != y.shape[0]:
                        continue
            #clamp parameters 
            #lambda_x.weight.data.clamp_(clamp_dowm, clamp_up)
            #lambda_x.bias.data.clamp_(clamp_down, clamp_up)

            #lambda_y.weight.data.clamp_(clamp_dowm, clamp_up)
            #lambda_y.bias.data.clamp_(clamp_down, clamp_up)
                for param_x, param_y in zip(lambda_x.parameters(), lambda_y.parameters()):
                        param_x.data.clamp_(-0.1, 0.1)
                        param_y.data.clamp_(-0.1, 0.1)

                z = Tensor(batch_size, latent_size, 1, 1).normal_(0, 1)#.to(device)
                x = x.expand_as(y)
                lambda_x.zero_grad()
                lambda_y.zero_grad()

                real_lambda_x = lambda_x(x)
                real_lambda_y = lambda_y(y)

                real_lambda_x.backward(one)
                real_lambda_y.backward(one)
            #print(Gx(z).data.shape)
                fake_lambda_x = lambda_x(Gx(z).data)
                fake_lambda_y = lambda_y(Gy(z).data)

                err_lambda_fake = fake_lambda_x + fake_lambda_y
                err_lambda_fake.backward(minus_one)
                err_lambda = real_lambda_x + real_lambda_y - err_lambda_fake

                optimizer_lambda.step()

            # или надо обновлять градиенты?/? who knows/ i think not
            # torch.nn.clip_grad_value_(lambda_x, clamp_up)

        # update generators parameters
        for param_x, param_y in zip(lambda_x.parameters(), lambda_y.parameters()):
            param_x.requires_grad = False
            param_y.requires_grad = False

        Gx.zero_grad()
        Gy.zero_grad()

        z = Tensor(batch_size, latent_size, 1, 1).normal_(0, 1)#.to(device)
        fake_x = Gx(z)
        fake_y = Gy(z)
        err_G = lambda_x(fake_x) + lambda_y(fake_y) + eta*c(fake_x, fake_y)
        err_G.backward(one)

        optimizer_G.step()

    if epoch % 100 == 0:
        print('Epoch f{}/{} || Loss_lambda: {:.4f} || Loss_G {:.4f}'.format(epoch, n_epoch, err_lambda.data[0], err_G.data[0]))

        x_cpu = x_cpu.mul(0.5).add(0.5)
        y_cpu = y_cpu.mul(0.5).add(0.5)
        fake_x = Gx(fixed_z).data.mul(0.5).add(0.5)
        fake_y = Gy(fixed_z).data.mul(0.5).add(0.5)

        save_image(x_cpu, './{}/real_samplex.png'.format(0))
        save_image(y_cpu, './{}/real_sampley.png'.format(0))
        save_image(fake_x, './{}/fake_samplex_{}.png'.format(0, epoch))
        save_image(fake_y, './{}/fake_samplex{}.png'.format(0, epoch))

        # do checkpointing
        torch.save(Gx.state_dict(), './0/Gx.pth')
        torch.save(Gy.state_dict(), './0/Gy.pth')
        torch.save(lambda_x.state_dict(), './0/lambda_x.pth')
        torch.save(lambda_y.state_dict(), './0/lambda_y.pth')


