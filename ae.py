import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self._name = 'Encoder'
        self.dim = args.edim
        self.conv1 = nn.Conv2d(1, self.dim*2, 3, stride=3, padding=1)
        self.conv2 = nn.Conv2d(self.dim*2, self.dim, 3, stride=2, padding=1)
        #self.conv3 = nn.Conv2d(2*self.dim, 4*self.dim, 5, stride=2, padding=2)
        #self.linear1 = nn.Linear(4*4*4*self.dim, self.dim)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=1)

    def forward(self, input):
        x = input.view(-1, 1, 28, 28)
        x = self.relu(self.conv1(input))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        #x = self.relu(self.dropout(self.conv3(x)))
        #x = x.view(-1, 4*4*4*self.dim)
        #x = self.linear1(x)
        #return x.view(-1, self.dim)
        return x

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self._name = 'Decoder'
        self.dim = args.ddim
        #self.linear1 = nn.Linear(self.dim, 4*4*4*self.dim)
        self.deconv1 = nn.ConvTranspose2d(self.dim, self.dim*2, 3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(self.dim*2, self.dim, 5, stride=3, padding=1)
        self.deconv_out = nn.ConvTranspose2d(self.dim, 1, 2, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)

    def forward(self, x):
        #x = self.relu(self.linear1(input))
        #x = x.view(-1, 4*self.dim, 4, 4)
        x = self.relu(self.deconv1(x))
        #x = x[:, :, :7, :7]
        x = self.relu(self.deconv2(x))
        x = self.deconv_out(x)
        x = self.sigmoid(x)
        return x.view(-1, 784)
