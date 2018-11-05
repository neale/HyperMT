import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
import utils


class Encoderz(nn.Module):
    def __init__(self, args):
        super(Encoderz, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Encoderz'
        self.linear1 = nn.Linear(self.ze, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, self.z*8)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #print ('E in: ', x.shape)
        x = x.view(-1, self.ze) #flatten filter size
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, 8, self.z)
        w1 = x[:, 0]
        w2 = x[:, 1]
        w3 = x[:, 2]
        w4 = x[:, 3]
        w5 = x[:, 4]
        w6 = x[:, 5]
        w7 = x[:, 6]
        w8 = x[:, 7]
        #print ('E out: ', x.shape)
        return w1, w2, w3, w4, w5, w6, w7, w8

""" Convolutional (1 x 16 x 5 x 5) """
class GeneratorE1(nn.Module):
    def __init__(self, args):
        super(GeneratorE1, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorE1'
        dim = self.edim
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 1*(dim*2)*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, self.edim*2, 1, 3, 3)
        #print ('W2 out: ', x.shape)
        return x

""" Convolutional (16 x 32 x 5 x 5) """
class GeneratorE2(nn.Module):
    def __init__(self, args):
        super(GeneratorE2, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorE2'
        dim = self.edim
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, dim*(2*dim)*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, self.edim, self.edim*2, 3, 3)
        #print ('W2 out: ', x.shape)
        return x

""" Convolutional (32 x 64 x 5 x 5) """
class GeneratorE3(nn.Module):
    def __init__(self, args):
        super(GeneratorE3, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorE3'
        dim = self.edim
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, (dim*2)*(dim*4)*5*5)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, self.edim*4, self.edim*2, 5, 5)
        #print ('W2 out: ', x.shape)
        return x

""" Linear (16 x 4096) """
class GeneratorE4(nn.Module):
    def __init__(self, args):
        super(GeneratorE4, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorE4'
        dim = self.edim
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, dim * (4*4*4*dim))
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, self.edim, 4*4*4*self.edim)
        #print ('W2 out: ', x.shape)
        return x


""" Linear (4096 x 16) """
class GeneratorD1(nn.Module):
    def __init__(self, args):
        super(GeneratorD1, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorD1'
        dim = self.ddim
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, dim * (4*4*4*dim))
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 4*4*4*self.ddim, self.ddim)
        #print ('W2 out: ', x.shape)
        return x


""" Convolutional (64 x 32 x 5 x 5) """
class GeneratorD2(nn.Module):
    def __init__(self, args):
        super(GeneratorD2, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorD2'
        dim = self.ddim
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, dim*(dim*2)*3*3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, self.ddim, self.ddim*2, 3, 3)
        #print ('W2 out: ', x.shape)
        return x

""" Convolutional (32 x 16 x 5 x 5) """
class GeneratorD3(nn.Module):
    def __init__(self, args):
        super(GeneratorD3, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorD3'
        dim = self.ddim
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, dim*(dim*2)*5*5)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, self.ddim*2, self.ddim, 5, 5)
        #print ('W2 out: ', x.shape)
        return x

""" Convolutional (16 x 1 x 5 x 5) """
class GeneratorD4(nn.Module):
    def __init__(self, args):
        super(GeneratorD4, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorD4'
        dim = self.ddim
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, dim*1*2*2)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, self.ddim, 1, 2, 2)
        #print ('W2 out: ', x.shape)
        return x


class DiscriminatorZ(nn.Module):
    def __init__(self, args):
        super(DiscriminatorZ, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        
        self.name = 'DiscriminatorZ'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print ('Dz in: ', x.shape)
        x = x.view(self.batch_size, -1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.sigmoid(x)
        # print ('Dz out: ', x.shape)
        return x
