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
        self.linear3 = nn.Linear(512, self.z*3)
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
        x = x.view(-1, 3, self.z)
        w1 = x[:, 0]
        w2 = x[:, 1]
        w3 = x[:, 2]
        #print ('E out: ', x.shape)
        return w1, w2, w3

""" Convolutional (1 x 16 x 5 x 5) """
class GeneratorE1(nn.Module):
    def __init__(self, args):
        super(GeneratorE1, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorE1'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 1*16*5*5)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 1, 16, 5, 5)
        #print ('W2 out: ', x.shape)
        return x

""" Convolutional (16 x 32 x 5 x 5) """
class GeneratorE2(nn.Module):
    def __init__(self, args):
        super(GeneratorE2, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorE2'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 16*32*5*5)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, 16, 32, 5, 5)
        #print ('W2 out: ', x.shape)
        return x

""" Convolutional (32 x 64 x 5 x 5) """
class GeneratorE3(nn.Module):
    def __init__(self, args):
        super(GeneratorE3, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorE3'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 32*64*5*5)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, 32, 64, 5, 5)
        #print ('W2 out: ', x.shape)
        return x

""" Linear (16 x 4096) """
class GeneratorE4(nn.Module):
    def __init__(self, args):
        super(GeneratorE4, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorE4'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 16 * 1024)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 16, 1024)
        #print ('W2 out: ', x.shape)
        return x


""" Linear (4096 x 16) """
class GeneratorD1(nn.Module):
    def __init__(self, args):
        super(GeneratorD1, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorD1'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 1024 * 16)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 1024, 16)
        #print ('W2 out: ', x.shape)
        return x


""" Convolutional (64 x 32 x 5 x 5) """
class GeneratorD2(nn.Module):
    def __init__(self, args):
        super(GeneratorD2, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorD2'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 64*32*5*5)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, 32, 64, 5, 5)
        #print ('W2 out: ', x.shape)
        return x

""" Convolutional (32 x 16 x 5 x 5) """
class GeneratorD2(nn.Module):
    def __init__(self, args):
        super(GeneratorD2, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorD2'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 16*32*5*5)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        x = x.view(-1, 16, 32, 5, 5)
        #print ('W2 out: ', x.shape)
        return x

""" Convolutional (16 x 1 x 5 x 5) """
class GeneratorD3(nn.Module):
    def __init__(self, args):
        super(GeneratorD3, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'GeneratorD3'
        self.linear1 = nn.Linear(self.z, 512)
        self.linear2 = nn.Linear(512, 16*1*8*8)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        #print ('W2 in: ', x.shape)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        x = x.view(-1, 16, 1, 8, 8)
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
