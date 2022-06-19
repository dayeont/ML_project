#!/usr/bin/env python
# coding: utf-8

# #### Imports

# In[1]:

from __future__ import print_function

import sys
import os
import urllib
import gzip
import pickle
import numpy as np
from os.path import dirname
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F




# In[2]:


# #### Variables and parameters

# In[3]:

# Sourse and target datasets can be set here manually

s = 'mnist'
t = 'usps'
root_dir = '~/dataset'

procs = 2
batchsize = 128 
tot_iter = 30000 # number of epochs 
z_dim = 100 # dimension of random noise
lrCl = 0.0002 # learning rate 
lrGen = 0.0002 # learning rate 
beta1 = 0.5 # beta1 for adam. 0.5)
beta2 = 0.999 # beta2 for adam
weight_decay =  0.0005 # weight_decay

t_iter = 500 # testiter


# #### Dataset preprocessing

# In[4]:


DataAttDict = {
'mnist': (1,28),
'mnistm': (3,28),
'usps': (1,28),
'svhn': (3,32),
}


# In[5]:


class Logger(object):
    def __init__(self, filepath = "./log.txt", mode = "w", stdout = None):
        if stdout==None:
            self.terminal = sys.stdout
        else:
            self.terminal = stdout
        os.makedirs(dirname(filepath), exist_ok=True)
        self.log = open(filepath, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        os.fsync(self.log)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

# def InfIter(_loader):
#     return iter(InfIter_C(_loader))

class InfIter:
    def __init__(self,_loader):
        self._loader = _loader
        self._iter = iter(_loader)
    def __iter__(self):
        return self
    def __next__(self):
        try:
            return self._iter.next()
        except StopIteration:
            self._iter = iter(self._loader)
            return self._iter.next()

##### define dataset
"""Dataset setting and data loader for MNIST-M.
Modified from
https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
CREDIT: https://github.com/corenel
"""
import errno
import os

import torch
import torch.utils.data as data
from PIL import Image


class MNISTM(data.Dataset):
    """`MNIST-M Dataset."""

    url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"

    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'mnist_m_train.pt'
    test_file = 'mnist_m_test.pt'

    def __init__(self,
                 root, mnist_root="data",
                 train=True,
                 transform=None, target_transform=None,
                 download=False):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__()
        self.root = os.path.expanduser(root)
        self.mnist_root = os.path.expanduser(mnist_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels =                 torch.load(os.path.join(self.root,
                                        self.processed_folder,
                                        self.training_file))
        else:
            self.test_data, self.test_labels =                 torch.load(os.path.join(self.root,
                                        self.processed_folder,
                                        self.test_file))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,
                                           self.processed_folder,
                                           self.training_file)) and \
            os.path.exists(os.path.join(self.root,
                                        self.processed_folder,
                                        self.test_file))

    def download(self):
        """Download the MNIST data."""
        # import essential packages
        from six.moves import urllib
        import gzip
        import pickle
        from torchvision import datasets

        # check if dataset already exists
        if self._check_exists():
            return

        # make data directories
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # download pkl files
        print('Downloading ' + self.url)
        filename = self.url.rpartition('/')[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.exists(file_path.replace('.gz', '')):
            data = urllib.request.urlopen(self.url)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f,                     gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        # load MNIST-M images from pkl file
        with open(file_path.replace('.gz', ''), "rb") as f:
            mnist_m_data = pickle.load(f, encoding='bytes')
        mnist_m_train_data = torch.ByteTensor(mnist_m_data[b'train'])
        mnist_m_test_data = torch.ByteTensor(mnist_m_data[b'test'])

        # get MNIST labels
        mnist_train_labels = datasets.MNIST(root=self.mnist_root,
                                            train=True,
                                            download=True).train_labels
        mnist_test_labels = datasets.MNIST(root=self.mnist_root,
                                           train=False,
                                           download=True).test_labels

        # save MNIST-M dataset
        training_set = (mnist_m_train_data, mnist_train_labels)
        test_set = (mnist_m_test_data, mnist_test_labels)
        with open(os.path.join(self.root,
                               self.processed_folder,
                               self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root,
                               self.processed_folder,
                               self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')


class USPS(data.Dataset):
    """USPS Dataset.
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=False):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        self.train_data = self.train_data.transpose(
            (0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        label = np.long(label)
        # label = torch.LongTensor([np.int64(label).item()])
        # label = torch.FloatTensor([label.item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels


# In[18]:


def createDataset(dataname, train):
        if dataname == "mnist":
            return dset.MNIST(root=root_dir+'/mnist', train=train, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]))
        if dataname == "mnistm":
            return MNISTM(root=root_dir+'/mnistm', mnist_root=dataroot+'/mnist', train=train, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]))
        if dataname == "usps":
            return USPS(root=root_dir+'/usps', train=train, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]))
        if dataname == "svhn":
            return dset.SVHN(root=root_dir+'/svhn', split=("train" if train else "test"), download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(28),
                                    transforms.ToTensor()
                                ]))


# #### Loss functions

# In[7]:


def loss(x):
    return (F.softplus(x)).mean()


# #### Neural net class

# In[33]:


class CoDis28x28(nn.Module):
    def __init__(self, ch_s, imsize_s, ch_t, imsize_t):
        super(CoDis28x28, self).__init__()
        self.conv0_s = nn.Conv2d(ch_s, 20, kernel_size=5, stride=1, padding=0)
        self.conv0_t = nn.Conv2d(ch_t, 20, kernel_size=5, stride=1, padding=0)
        self.pool0 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(50, 500, kernel_size=4, stride=1, padding=0)
        self.prelu2 = nn.PReLU()
        self.conv30_s = nn.Conv2d(500, 100, kernel_size=1, stride=1, padding=0)
        self.prelu3_s = nn.PReLU()
        self.conv31_s = nn.Conv2d(100, 1, kernel_size=1, stride=1, padding=0)
        self.conv30_t = nn.Conv2d(500, 100, kernel_size=1, stride=1, padding=0)
        self.prelu3_t = nn.PReLU()
        self.conv31_t = nn.Conv2d(100, 1, kernel_size=1, stride=1, padding=0)
        self.conv_cl = nn.Conv2d(500, 10, kernel_size=1, stride=1, padding=0)

    def forward(self, x_s, x_t):
        h0_s = self.pool0(self.conv0_s(x_s))
        h0_t = self.pool0(self.conv0_t(x_t))
        h1_s = self.pool1(self.conv1(h0_s))
        h1_t = self.pool1(self.conv1(h0_t))
        h2_s = self.prelu2(self.conv2(h1_s))
        h2_t = self.prelu2(self.conv2(h1_t))
        h3_s = self.conv31_s(self.prelu3_s(self.conv30_s(h2_s)))
        h3_t = self.conv31_t(self.prelu3_t(self.conv30_t(h2_t)))
        return h3_s, h2_s, h0_s, h3_t, h2_t, h0_t

    def pred_s(self, x_s):
        h0_s = self.pool0(self.conv0_s(x_s))
        h1_s = self.pool1(self.conv1(h0_s))
        h2_s = self.prelu2(self.conv2(h1_s))
        h3_s = self.conv_cl(h2_s)
        return h3_s.squeeze(), h2_s.squeeze()

    def pred_t(self, x_t):
        h0_t = self.pool0(self.conv0_t(x_t))
        h1_t = self.pool1(self.conv1(h0_t))
        h2_t = self.prelu2(self.conv2(h1_t))
        h3_t = self.conv_cl(h2_t)
        return h3_t.squeeze(), h2_t.squeeze()

    def pred_fromrep(self, h2):
        return self.conv_cl(h2).squeeze()


# Generator Model
class CoGen28x28(nn.Module):
    def __init__(self, ch_s, imsize_s, ch_t, imsize_t, zsize):
        super(CoGen28x28, self).__init__()
        self.dconv0 = nn.ConvTranspose2d(zsize, 1024, kernel_size=4, stride=1)
        self.bn0 = nn.BatchNorm2d(1024, affine=False)
        self.prelu0 = nn.PReLU()
        self.dconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512, affine=False)
        self.prelu1 = nn.PReLU()
        self.dconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256, affine=False)
        self.prelu2 = nn.PReLU()
        self.dconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128, affine=False)
        self.prelu3 = nn.PReLU()
        self.dconv4_s = nn.ConvTranspose2d(128, ch_s, kernel_size=6, stride=1, padding=1)
        self.dconv4_t = nn.ConvTranspose2d(128, ch_t, kernel_size=6, stride=1, padding=1)
        self.sig4_s = nn.Sigmoid()
        self.sig4_t = nn.Sigmoid()

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        h0 = self.prelu0(self.bn0(self.dconv0(z)))
        h1 = self.prelu1(self.bn1(self.dconv1(h0)))
        h2 = self.prelu2(self.bn2(self.dconv2(h1)))
        h3 = self.prelu3(self.bn3(self.dconv3(h2)))
        out_s = self.sig4_s(self.dconv4_s(h3))
        out_t = self.sig4_t(self.dconv4_t(h3))
        return out_s, out_t


# #### To be determined

# In[9]:


def xavier_weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0.1)

class C_contour(nn.Module):
    def __init__(self):
        super(C_contour, self).__init__()
        self.L1 = nn.Conv2d(3, 1, 3, bias=False)
        C1 = np.array([[-1,0,1],[-2,0,2], [-1,0,1]])
        C1 = torch.from_numpy(C1)
        list(self.L1.parameters())[0].data[0,:,:,:] = C1.unsqueeze(0)
        list(self.L1.parameters())[0].requires_grad = False

        self.L2 = nn.Conv2d(3, 1, 3, bias=False)
        C2 = np.array([[1,2,1],[0,0,0], [-1,-2,-1]])
        C2 = torch.from_numpy(C2)
        list(self.L2.parameters())[0].data[0,:,:,:] = C2.unsqueeze(0)
        list(self.L2.parameters())[0].requires_grad = False

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = torch.cat([x,x,x],1)
        if y.shape[1] == 1:
            y = torch.cat([y,y,y],1)
        imgx1 = self.L1(x)
        imgx2 = self.L2(x)

        imgy1 = self.L1(y)
        imgy2 = self.L2(y)
#       img = img.view(img.shape[0], *img_shape)
        return torch.norm(imgx1-imgy1)+torch.norm(imgx2-imgy2)


# ##### Test function

# In[10]:


def test_f(verbose = True, print_period = 100):
        # VALIDATION
        j = 0
        cum_acc = 0
        total_len = 0

        netCl.eval()
        for y, y_label in dataloader_y:
            j = j+1

            y = y.to(device)
            y_label = y_label.to(device)

            # compute output
            outputs, _ = netCl.pred_t(y)
            test_loss = criterion(outputs, y_label)

            pred = torch.argmax(outputs,dim=-1)
            test_acc = torch.sum(pred==y_label).item()
            cum_acc = cum_acc+test_acc
            test_acc = test_acc/len(pred)
            total_len += len(pred)
            if j%print_period==0 and verbose:
                print('Iter: [%d/%d],  Test Loss:  %.8f, Test Acc:  %.2f' % (j,len(dataloader_y),test_loss, test_acc))
        print(' Test acc for the epoch:  %.8f\n##############################################' % (cum_acc/total_len))
        return cum_acc/total_len


# ##### Visualization and learning rate update

# In[53]:


def show_tsne(xr, xl, yr, yl, xfr, xfl, yfr, yfl, epoch):
        import sklearn
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=0)
        X = np.concatenate((xr, yr, xfr, yfr), axis=0)
        X_2d = tsne.fit_transform(X)
        from matplotlib import pyplot as plt
        plt.figure(figsize=(6, 5))
        colors = np.array(['r', 'g', 'b', 'c', 'm', 'y', 'k', 'grey', 'orange', 'purple'])
        plt.scatter(X_2d[:batchsize, 0], X_2d[:batchsize, 1], c=colors[xl], marker="o", label=["source"])
        for i in range(batchsize):
            plt.text(X_2d[batchsize+i, 0], X_2d[batchsize+i, 1], str(yl[i]), color=colors[yl[i]], label="target")
        plt.scatter(X_2d[batchsize:batchsize*2, 0], X_2d[batchsize:batchsize*2, 1], c=colors[yl], marker="*", label=["target"])
        plt.scatter(X_2d[batchsize*2:batchsize*3, 0], X_2d[batchsize*2:batchsize*3, 1], marker="_", c=colors[xfl], label="source fake")
        plt.scatter(X_2d[batchsize*3:batchsize*4, 0], X_2d[batchsize*3:batchsize*4, 1], marker="+", c=colors[yfl], label="target fake")
        plt.legend()
        plt.savefig(experiment +'/tsne_%05d.pdf'%(epoch), bbox_inches='tight',format="pdf", dpi = 300)
        plt.close()



# ## Main

# ##### Directories organization

# In[13]:



name = s + "two" + t
experiment = "Experiment_DASPOT/" + name
os.system('mkdir {0}'.format(experiment))
stdout_backup = sys.stdout
sys.stdout = Logger(experiment +"/log.txt","w", stdout_backup)
manualseed = random.randint(1, 10000) 

random.seed(manualseed)
torch.manual_seed(manualseed)


# ##### Cuda

# In[14]:


if torch.cuda.is_available():
        device = torch.device("cuda:0")
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
else:
    device = torch.device("cpu")


# ##### Dataset loaders

# In[19]:


dataset1 = createDataset(s, True)
dataset2 = createDataset(t, True)
dataset3 = createDataset(t, False)

dataloader_x = torch.utils.data.DataLoader(dataset1, batch_size=batchsize,
                                            shuffle=True, num_workers=int(procs), pin_memory=True, drop_last = True)
dataloader_y_ans = torch.utils.data.DataLoader(dataset2, batch_size=batchsize,
                                            shuffle=True, num_workers=int(procs), pin_memory=True, drop_last = True)
dataloader_y = torch.utils.data.DataLoader(dataset3, batch_size=batchsize,
                                            shuffle=True, num_workers=int(procs), pin_memory=True)


# ##### Nets

# In[21]:


netCl = CoDis28x28(*DataAttDict[s],*DataAttDict[t]).to(device)
netD_1 = nn.Conv2d(20, 1, kernel_size=12, stride=1, padding=0).to(device)
netD_2 = nn.Sequential(
                    nn.Conv2d(500, 100, kernel_size=1, stride=1, padding=0),
                    nn.PReLU(),
                    nn.Conv2d(100, 1, kernel_size=1, stride=1, padding=0)
                         ).to(device)
netGen = CoGen28x28(*DataAttDict[s],*DataAttDict[t], zsize=z_dim).to(device)


# ##### Optimizer setup

# In[26]:


optimizerCl = optim.Adam([p for p in netCl.parameters() if p.requires_grad], lr=lrCl, betas=(beta1, beta2), weight_decay=weight_decay)
optimizerD = optim.Adam([p for p in netD_1.parameters() if p.requires_grad]+[p for p in netD_2.parameters() if p.requires_grad], lr=lrCl, betas=(beta1, beta2), weight_decay=weight_decay)
optimizerGen = optim.Adam([p for p in netGen.parameters() if p.requires_grad], lr=lrGen, betas=(beta1, beta2), weight_decay=weight_decay)


# ##### Loss criterion 

# In[27]:


criterion = nn.CrossEntropyLoss()
c_loss = C_contour().to(device)


# ##### GAN training

# In[ ]:


best_test_acc = 0
x_noise = torch.randn(batchsize, z_dim).to(device)
fixed_noise = x_noise
fixed_x = None
fixed_y = None


y_iter = InfIter(dataloader_y_ans)
x_iter = InfIter(dataloader_x)
for i in range(tot_iter):
    netCl.train()
    netD_1.train()
    netD_2.train()
    netGen.train()
    for in_iter in range(2):
        netCl.zero_grad()
        netD_1.zero_grad()
        netD_2.zero_grad()

        y, y_labels = next(y_iter)
        x, x_labels = next(x_iter)
        z = torch.randn(batchsize, z_dim).to(device)

        x = x.to(device)
        y = y.to(device)
        x_labels = x_labels.to(device)

            # GAN training
        x_f, y_f = netGen(z)
        x_3,x_2,x_1, y_3,y_2,y_1 = netCl(x,y)
        x_3_f,x_2_f,_, y_3_f,y_2_f,_ = netCl(x_f.detach(), y_f.detach())

        errD_xy = loss(netD_1(x_1.detach())) 
        errD_xy += loss(-netD_1(y_1.detach())) 
        errD_xy += loss(netD_2(x_2.detach())) 
        errD_xy += loss(-netD_2(y_2.detach())) 
        errD_xy.backward()
        optimizerD.step()

        errD_xy = loss(-netD_1(x_1)) 
        errD_xy += loss(netD_1(y_1)) 
        errD_xy += loss(-netD_2(x_2.detach())) 
        errD_xy += loss(netD_2(y_2.detach())) 


        errD_x_real = loss(x_3) 
        errD_y_real = loss(y_3) 
        errD_x_fake = loss(-x_3_f) 
        errD_y_fake = loss(-y_3_f) 
        D_x_real = x_3.mean().item()
        D_y_real = y_3.mean().item()
        D_x_fake = x_3_f.mean().item()
        D_y_fake = y_3_f.mean().item()

        x_out = netCl.pred_fromrep(x_2)
        #netCl.eval()
        x_out_f = netCl.pred_fromrep(x_2_f)
        #netCl.train()

        x_prob_fake = F.softmax(x_out_f, dim=1)
        x_maxprob_fake,x_label_fake = x_prob_fake.max(dim=1)
        select_indices = x_maxprob_fake>0.9
        ys_rep_fake = y_2_f[select_indices,:,:,:]
        if(ys_rep_fake.shape[0]==0):
            errCl_x = 0
        else:
            ys_label = x_label_fake[select_indices].detach()
            ys_out_fake = netCl.pred_fromrep(ys_rep_fake)
            if ys_rep_fake.shape[0]==1:
                ys_out_fake = ys_out_fake[None, :]
            errCl_x = criterion(ys_out_fake,ys_label) 


        optloss = ((x_2_f-y_2_f)**2).sum()/batchsize 
        errCl_x += criterion(x_out,x_labels) 
            # GAN training for y
        lossCl = errD_x_real+errD_y_real+errD_x_fake+errD_y_fake+optloss+errCl_x+errD_xy
        lossCl.backward()
        optimizerCl.step()
        
    netGen.zero_grad()

    x_f, y_f = netGen(z)
    x_3_f,x_2_f,_,y_3_f,y_2_f,_ = netCl(x_f, y_f)
    errD_x_fake = loss(x_3_f) 
    errD_y_fake = loss(y_3_f) 
    D_x_fake = x_3_f.mean().item()
    D_y_fake = y_3_f.mean().item()

        # train optimal transport loss
    optloss = ((x_2_f-y_2_f)**2).sum()/batchsize #+ opt_contour_loss(x_fake,y_fake) * OPTlossscale2

        # Total Loss
    total_loss = errD_x_fake+errD_y_fake+optloss
    total_loss.backward()

    optimizerGen.step()

    pred = torch.argmax(x_out,dim=-1)
    train_acc = torch.sum(pred==x_labels).item()/len(pred)

    if i%100==0:
        print('Iter: [%d/%d] D_x_real: %.4f, D_x_fake: %.4f, D_y_real: %.4f, D_y_fake: %.4f, Loss_GANx: %.4f, Loss_GANy: %.4f, Loss_OPT: %.4f, Loss_P: %.4f, Train Accu: %.4f' %
            (i, tot_iter, D_x_real, D_x_fake, D_y_real, D_y_fake, errD_x_real.item()+errD_x_fake.item(), errD_y_real.item()+errD_y_fake.item(), optloss.item(), errCl_x.item(), train_acc))

    netCl.eval()
        # show tsne
    if i%t_iter == 0:
        if fixed_x is None:
            fixed_x = x.clone()
            fixed_y = y.clone()
            fixed_xlabel = x_labels.to("cpu").long().numpy()
            fixed_ylabel = y_labels.to("cpu").long().numpy()
            if fixed_x.shape[1] == fixed_y.shape[1]:
                real_images = torch.cat((fixed_x, fixed_y), 2)
            elif fixed_x.shape[1] == 1:
                real_images = torch.cat((torch.cat((fixed_x,fixed_x,fixed_x), 1), fixed_y), 2)
            else:
                real_images = torch.cat((fixed_x, torch.cat((fixed_y,fixed_y,fixed_y), 1)), 2)
            torchvision.utils.save_image(real_images.data, experiment +'/realimage.jpg')
        _,fixedx_rep = netCl.pred_s(fixed_x)
        _,fixedy_rep = netCl.pred_t(fixed_y)
        fixed_x_fake, fixed_y_fake = netGen(fixed_noise)
        fixedx_rep_fake_l,fixedx_rep_fake = netCl.pred_s(fixed_x_fake)
        fixedx_rep_fake_l = fixedx_rep_fake_l.argmax(dim=1)
        fixedy_rep_fake_l,fixedy_rep_fake = netCl.pred_t(fixed_y_fake)
        fixedy_rep_fake_l = fixedy_rep_fake_l.argmax(dim=1)
        if fixed_x_fake.shape[1] == fixed_y_fake.shape[1]:
            fake_images = torch.cat((fixed_x_fake, fixed_y_fake), 2)
        elif fixed_x_fake.shape[1] == 1:
            fake_images = torch.cat((torch.cat((fixed_x_fake,fixed_x_fake,fixed_x_fake), 1), fixed_y_fake), 2)
        else:
            fake_images = torch.cat((fixed_x_fake, torch.cat((fixed_y_fake,fixed_y_fake,fixed_y_fake), 1)), 2)
        torchvision.utils.save_image(fake_images.data, experiment +'/fakeimage_%05d.jpg'%(i))
        show_tsne(
            fixedx_rep.to("cpu").detach().numpy(),
            fixed_xlabel,
            fixedy_rep.to("cpu").detach().numpy(),
            fixed_ylabel,
            fixedx_rep_fake.to("cpu").detach().numpy(),
            fixedx_rep_fake_l,
            fixedy_rep_fake.to("cpu").detach().numpy(),
            fixedy_rep_fake_l,
            i)







