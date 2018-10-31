import os
import sys
import time
import argparse
import numpy as np
from glob import glob
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch import autograd
from torch import optim
from torch.nn import functional as F
import torch.distributions.multivariate_normal as N
import pprint

import models.twinmodels as models
import ops
import utils
import netdef
import datagen


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('-z', '--z', default=512, type=int, help='latent space width')
    parser.add_argument('-ze', '--ze', default=512, type=int, help='encoder dimension')
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-e', '--epochs', default=200000, type=int)
    parser.add_argument('-s', '--model', default='mednet', type=str)
    parser.add_argument('-d', '--dataset', default='cifar', type=str)
    parser.add_argument('--beta', default=1., type=float)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--use_x', default=False, type=bool, help='sample from real layers')
    parser.add_argument('--pretrain_e', default=False, type=bool)
    parser.add_argument('--scratch', default=False, type=bool)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--hidden', default=False, type=bool)
    args = parser.parse_args()
    return args


def sample_z(args, grad=True):
    z = torch.randn(args.batch_size, args.dim, requires_grad=grad).cuda()
    return z


def sample_z_like(shape, scale=1., grad=True):
    mean = torch.zeros(shape[1])
    cov = torch.eye(shape[1])
    D = N.MultivariateNormal(mean, cov)
    z = D.sample((shape[0],)).cuda()
    return scale * z
 

# hard code the two layer net
def train_clf(args, Z, data, target):
    """ calc classifier loss """
    data, target = data.cuda(), target.cuda()
    x = F.relu(F.conv2d(data, Z[0]))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(F.conv2d(x, Z[1]))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(F.conv2d(x, Z[2]))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(x.size(0), -1)
    x = F.relu(F.linear(x, Z[3]))
    x = F.linear(x, Z[4])
    loss = F.cross_entropy(x, target)
    pred = x.data.max(1, keepdim=True)[1]
    correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
    return (correct, loss)


def batch_zero_grad(nets):
    for module in nets:
        module.zero_grad()


def batch_update_optim(optimizers):
    for optim in optimizers:
        optim.step()

def free_params(nets):
    for module in nets:
        for p in module.parameters():
            p.requires_grad = True


def frozen_params(nets):
    for module in nets:
        for p in module.parameters():
            p.requires_grad = False


def pretrain_loss(encoded, noise):
    mean_z = torch.mean(noise, dim=0, keepdim=True)
    mean_e = torch.mean(encoded, dim=0, keepdim=True)
    mean_loss = F.mse_loss(mean_z, mean_e)

    cov_z = torch.matmul((noise-mean_z).transpose(0, 1), noise-mean_z)
    cov_z /= 999
    cov_e = torch.matmul((encoded-mean_e).transpose(0, 1), encoded-mean_e)
    cov_e /= 999
    cov_loss = F.mse_loss(cov_z, cov_e)
    return mean_loss, cov_loss


def train(args):
    
    torch.manual_seed(8734)
    
    netE = models.Encoder(args).cuda()
    W1 = models.GeneratorW1(args).cuda()
    W2 = models.GeneratorW2(args).cuda()
    W3 = models.GeneratorW3(args).cuda()
    W4 = models.GeneratorW4(args).cuda()
    W5 = models.GeneratorW5(args).cuda()
    netD = models.DiscriminatorZ(args).cuda()
    print (netE, W1, W2, W3, W4, W5, netD)

    optimE = optim.Adam(netE.parameters(), lr=5e-3, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW1 = optim.Adam(W1.parameters(), lr=5e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW2 = optim.Adam(W2.parameters(), lr=5e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW3 = optim.Adam(W3.parameters(), lr=5e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW4 = optim.Adam(W4.parameters(), lr=5e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW5 = optim.Adam(W5.parameters(), lr=5e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=5e-5, betas=(0.5, 0.9), weight_decay=1e-4)
    
    m_best_test_acc, m_best_test_loss = 0., np.inf
    c_best_test_acc, c_best_test_loss = 0., np.inf
    args.m_best_loss, args.m_best_acc = m_best_test_loss, m_best_test_acc
    args.c_best_loss, args.c_best_acc = c_best_test_loss, c_best_test_acc
    
    mnist_train, mnist_test = datagen.load_mnist(args)
    cifar_train, cifar_test = datagen.load_cifar(args)
    x_dist = utils.create_d(args.ze)
    z_dist = utils.create_d(args.z)
    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()
    print ("==> pretraining encoder")
    j = 0
    final = 100.
    e_batch_size = 1000
    if args.pretrain_e:
        mask1 = torch.zeros(e_batch_size, args.ze).cuda()
        mask2 = torch.ones(e_batch_size, args.ze).cuda()
        for j in range(500):
            x = utils.sample_d(x_dist, e_batch_size)
            z = utils.sample_d(z_dist, e_batch_size)
            if j % 2 == 0: x = torch.cat((x, mask1), dim=0)
            if j % 2 == 1: x = torch.cat((x, mask2), dim=0)
            codes = netE(x)
            for i, code in enumerate(codes):
                code = code.view(e_batch_size, args.z)
                mean_loss, cov_loss = pretrain_loss(code, z)
                loss = mean_loss + cov_loss
                loss.backward(retain_graph=True)
            optimE.step()
            netE.zero_grad()
            print ('Pretrain Enc iter: {}, Mean Loss: {}, Cov Loss: {}'.format(
                j, mean_loss.item(), cov_loss.item()))
            final = loss.item()
            if loss.item() < 0.1:
                print ('Finished Pretraining Encoder')
                break

    print ('==> Begin Training')
    for _ in range(args.epochs):
        for batch_idx, (mnist, cifar) in enumerate(zip(mnist_train, cifar_train)):
            if batch_idx % 2 == 0:
                data, target = mnist
                mask = torch.zeros(args.batch_size, args.ze).cuda()
            else:
                data, target = cifar
                mask = torch.ones(args.batch_size, args.ze).cuda()

            batch_zero_grad([netE, W1, W2, W3, W4, W5, netD])
            z = utils.sample_d(x_dist, args.batch_size)
            z = torch.cat((z, mask), dim=0)
            codes = netE(z)
            l1 = W1(codes[0])
            l2 = W2(codes[1])
            l3 = W3(codes[2])
            l4 = W4(codes[3])
            l5 = W5(codes[4])
            # Z Adversary 
            for code in codes:
                noise = utils.sample_d(z_dist, args.batch_size)
                d_real = netD(noise)
                d_fake = netD(code)
                d_real_loss = -1 * torch.log((1-d_real).mean())
                d_fake_loss = -1 * torch.log(d_fake.mean())
                d_real_loss.backward(retain_graph=True)
                d_fake_loss.backward(retain_graph=True)
                d_loss = d_real_loss + d_fake_loss
            optimD.step()
            # Generator (Mean test)
            netD.zero_grad()
            z = utils.sample_d(x_dist, args.batch_size)
            z = torch.cat((z, mask), dim=0)
            codes = netE(z)
            l1 = W1(codes[0])
            l2 = W2(codes[1])
            l3 = W3(codes[2])
            l4 = W4(codes[3])
            l5 = W5(codes[4])
            d_real = []
            for code in codes:
                d = netD(code)
                d_real.append(d)
            netD.zero_grad()
            d_loss = torch.stack(d_real).log().mean() * 10.
            for (g1, g2, g3, g4, g5) in zip(l1, l2, l3, l4, l5):
                correct, loss = train_clf(args, [g1, g2, g3, g4, g5], data, target)
                scaled_loss = args.beta * loss
                if loss != loss:
                    sys.exit(0)
                scaled_loss.backward(retain_graph=True)
                d_loss.backward(torch.tensor(-1, dtype=torch.float).cuda(),retain_graph=True)
            optimE.step(); optimW1.step(); optimW2.step()
            optimW3.step(); optimW4.step(); optimW5.step()

            loss = loss.item()
            """ Update Statistics """
            if batch_idx % 50 == 0 or batch_idx % 50 == 1:
                acc = correct 
                print ('**************************************')
                if batch_idx % 50 == 0:
                    print ('MNIST Test: Enc, Dz, Lscale: {} test'.format(args.beta))
                if batch_idx % 50 == 1:
                    print ('CIFAR Test: Enc, Dz, Lscale: {} test'.format(args.beta))
                print ('Acc: {}, G Loss: {}, D Loss: {}'.format(acc, loss, d_loss))
                print ('best test loss: {}, {}'.format(args.m_best_loss, args.c_best_loss))
                print ('best test acc: {}, {}'.format(args.m_best_acc, args.c_best_acc))
                print ('**************************************')
            if batch_idx > 1 and batch_idx % 100 == 0:
                m_test_acc = 0.
                m_test_loss = 0.
                for i, (data, y) in enumerate(mnist_test):
                    z = utils.sample_d(x_dist, args.batch_size)
                    z = torch.cat((z, torch.zeros(args.batch_size, args.ze).cuda()), dim=0)
                    w1_code, w2_code, w3_code, w4_code, w5_code = netE(z)
                    l1 = W1(w1_code)
                    l2 = W2(w2_code)
                    l3 = W3(w3_code)
                    l4 = W4(w4_code)
                    l5 = W5(w5_code)
                    for (g1, g2, g3, g4, g5) in zip(l1, l2, l3, l4, l5):
                        correct, loss = train_clf(args, [g1, g2, g3, g4, g5], data, y)
                        m_test_acc += correct.item()
                        m_test_loss += loss.item()
                m_test_loss /= len(mnist_test.dataset) * args.batch_size
                m_test_acc /= len(mnist_test.dataset) * args.batch_size
                print ('MNIST Test Accuracy: {}, Test Loss: {}'.format(m_test_acc, m_test_loss))
                
                c_test_acc = 0.
                c_test_loss = 0
                for i, (data, y) in enumerate(cifar_test):
                    z = utils.sample_d(x_dist, args.batch_size)
                    z = torch.cat((z, torch.ones(args.batch_size, args.ze).cuda()), dim=0)
                    w1_code, w2_code, w3_code, w4_code, w5_code = netE(z)
                    l1 = W1(w1_code)
                    l2 = W2(w2_code)
                    l3 = W3(w3_code)
                    l4 = W4(w4_code)
                    l5 = W5(w5_code)
                    for (g1, g2, g3, g4, g5) in zip(l1, l2, l3, l4, l5):
                        correct, loss = train_clf(args, [g1, g2, g3, g4, g5], data, y)
                        c_test_acc += correct.item()
                        c_test_loss += loss.item()
                c_test_loss /= len(cifar_test.dataset) * args.batch_size
                c_test_acc /= len(cifar_test.dataset) * args.batch_size
                print ('CIFAR Test Accuracy: {}, Test Loss: {}'.format(c_test_acc, c_test_loss))

                if m_test_loss < m_best_test_loss or m_test_acc > m_best_test_acc:
                    #utils.save_hypernet_cifar(args, [netE, W1, W2, W3, W4, W5, netD], test_acc)
                    print ('==> new best stats, saving')
                    if m_test_loss < m_best_test_loss:
                        m_best_test_loss = m_test_loss
                        args.m_best_loss = m_test_loss
                    if m_test_acc > m_best_test_acc:
                        m_best_test_acc = m_test_acc
                        args.m_best_acc = m_test_acc

                if c_test_loss < c_best_test_loss or c_test_acc > c_best_test_acc:
                    #utils.save_hypernet_cifar(args, [netE, W1, W2, W3, W4, W5, netD], test_acc)
                    print ('==> new best stats, saving')
                    if c_test_loss < c_best_test_loss:
                        c_best_test_loss = c_test_loss
                        args.c_best_loss = c_test_loss
                    if c_test_acc > c_best_test_acc:
                        c_best_test_acc = c_test_acc
                        args.c_best_acc = c_test_acc



if __name__ == '__main__':

    args = load_args()
    modeldef = netdef.nets()[args.model]
    pprint.pprint (modeldef)
    # log some of the netstat quantities so we don't subscript everywhere
    args.stat = modeldef
    args.shapes = modeldef['shapes']
    train(args)
