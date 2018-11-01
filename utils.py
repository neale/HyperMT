import os
import sys
import time
import torch
import natsort
import datagen
import argparse
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import itertools
import cv2
import numpy as np

from glob import glob
from scipy.misc import imsave
import torch.nn as nn
import torch.nn.init as init
import torch.distributions.multivariate_normal as N


def sample_z(args, grad=True):
    z = torch.randn(args.batch_size, args.dim, requires_grad=grad).cuda()
    return z


def create_d(shape):
    mean = torch.zeros(shape)
    cov = torch.eye(shape)
    D = N.MultivariateNormal(mean, cov)
    return D


def sample_d(D, shape, scale=1., grad=True):
    z = scale * D.sample((shape,)).cuda()
    z.requires_grad = grad
    return z


def sample_z_like(shape, scale=1., grad=True):
    return torch.randn(*shape, requires_grad=grad).cuda()


def save_model(args, model, op):
    path = '{}/{}/{}_{}.pt'.format(
            args.dataset, args.model, model.name, args.exp)
    path = model_dir + path
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': op.state_dict(),
        'best_acc': args.best_acc,
        'best_loss': args.best_loss
        }, path)


def load_model(args, model, op):
    path = '{}/{}/{}_{}.pt'.format(
            args.dataset, args.model, model.name, args.exp)
    path = model_dir + path
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['state_dict'])
    op.load_state_dict(ckpt['optimizer'])
    acc = ckpt['best_acc']
    loss = ckpt['best_loss']
    return model, op, (acc, loss)


def get_net_only(model):
    net_dict = {
            'state_dict': model.state_dict(),
    }
    return net_dict


def load_net_only(model, d):
    model.load_state_dict(d['state_dict'])
    return model


# this is also hard coded right now to the specific model
# dont sue me 
def save_clf(args, Z, acc):
    """ gross """
    if args.dataset == 'mnist':
        import models.mnist_clf as models
        model = models.Small2().cuda()
    elif args.dataset == 'cifar':
        import models.cifar_clf as models
        model = models.MedNet().cuda() 
    """ end gross """

    state = model.state_dict()
    layers = zip(args.stat['layer_names'], Z)
    for i, (name, params) in enumerate(layers):
        name = name + '.weight'
        loader = state[name]
        state[name] = params.detach()
        assert state[name].equal(loader) == False
        model.load_state_dict(state)
    #import cifar
    #ac, loss = cifar.test(args, model, 0)
    #print ('acc: {}, loss: {}'.format(ac, loss))
    path = 'exp_models/hyper{}_clf_{}_{}.pt'.format(args.dataset, args.exp, acc)
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/HyperGAN/' + path
    print ('saving hypernet to {}'.format(path))
    torch.save({'state_dict': model.state_dict()}, path)


def save_hypernet_mnist(args, models, acc):
    netE, W1, W2, W3 = models
    hypernet_dict = {
            'E':  get_net_only(netE),
            'W1': get_net_only(W1),
            'W2': get_net_only(W2),
            'W3': get_net_only(W3),
            }
    path = 'exp_models/hypermnist_{}_{}.pt'.format(args.exp, acc)
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/HyperGAN/' + path
    torch.save(hypernet_dict, path)
    print ('Hypernet saved to {}'.format(path))


def save_hypernet_cifar(args, models, acc):
    netE, W1, W2, W3, W4, W5, netD = models
    hypernet_dict = {
            'E':  get_net_only(netE),
            'W1': get_net_only(W1),
            'W2': get_net_only(W2),
            'W3': get_net_only(W3),
            'W4': get_net_only(W4),
            'W5': get_net_only(W5),
            'D': get_net_only(netD),
            }
    path = 'exp_models/hypercifar_{}_{}.pt'.format(args.exp, acc)
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/HyperGAN/' + path
    torch.save(hypernet_dict, path)
    print ('Hypernet saved to {}'.format(path))


def save_hypernet_regression(args, models, mse):
    netE, W1, W2 = models
    hypernet_dict = {
            'E':  get_net_only(netE),
            'W1': get_net_only(W1),
            'W2': get_net_only(W2),
            }
    path = 'exp_models/hypertoy{}_{}.pt'.format(args.exp, mse)
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/HyperGAN/' + path
    torch.save(hypernet_dict, path)
    print ('Hypernet saved to {}'.format(path))


""" hard coded for mnist experiment dont use generally """
def load_hypernet(path, args=None):
    if args is None:
        args = load_default_args()
    #import models.models_mnist_small as hyper
    netE = hyper.Encoder(args).cuda()
    W1 = hyper.GeneratorW1(args).cuda()
    W2 = hyper.GeneratorW2(args).cuda()
    W3 = hyper.GeneratorW3(args).cuda()
    print ('loading hypernet from {}'.format(path))
    d = torch.load(path)
    netE = load_net_only(netE, d['E'])
    W1 = load_net_only(W1, d['W1'])
    W2 = load_net_only(W2, d['W2'])
    W3 = load_net_only(W3, d['W3'])
    return (netE, W1, W2, W3)


def sample_hypernet(hypernet, args=None):
    netE, W1, W2, W3 = hypernet
    x_dist = create_d(256)
    z = sample_d(x_dist, 32)
    #z = torch.randn(32, 300).cuda()
    codes = netE(z)
    l1 = W1(codes[0])
    l2 = W2(codes[1])
    l3 = W3(codes[2])
    return l1, l2, l3


def load_hypernet_cifar(path, args=None):
    if args is None:
        args = load_default_args()
    #import models.models_cifar_small as hyper
    netE = hyper.Encoder(args).cuda()
    W1 = hyper.GeneratorW1(args).cuda()
    W2 = hyper.GeneratorW2(args).cuda()
    W3 = hyper.GeneratorW3(args).cuda()
    W4 = hyper.GeneratorW4(args).cuda()
    W5 = hyper.GeneratorW5(args).cuda()
    print ('loading hypernet from {}'.format(path))
    d = torch.load(path)
    netE = load_net_only(netE, d['E'])
    W1 = load_net_only(W1, d['W1'])
    W2 = load_net_only(W2, d['W2'])
    W3 = load_net_only(W3, d['W3'])
    W4 = load_net_only(W4, d['W4'])
    W5 = load_net_only(W5, d['W5'])
    return (netE, W1, W2, W3, W4, W5)


def sample_hypernet_cifar(hypernet, args=None):
    netE, W1, W2, W3, W4, W5 = hypernet
    x_dist = create_d(512)
    z = sample_d(x_dist, 32)
    #z = torch.randn(32, 300).cuda()
    codes = netE(z)
    l1 = W1(codes[0])
    l2 = W2(codes[1])
    l3 = W3(codes[2])
    l4 = W4(codes[3])
    l5 = W5(codes[4])
    return l1, l2, l3, l4, l5


def weights_to_clf(weights, model, names):
    state = model.state_dict()
    layers = zip(names, weights)
    for i, (name, params) in enumerate(layers):
        name = name + '.weight'
        loader = state[name]
        state[name] = params.detach()
        model.load_state_dict(state)
    return model


def generate_image(args, iter, E, D, imgs, save_path=None):
    if save_path is None:
        if args.scratch:
            save_path = '/scratch/eecs-share/ratzlafn/imgs'
        else:
            save_path = './'
    batch_size = args.batch_size
    datashape = (1, 28, 28)
    samples = D(E(imgs))
    samples = samples.view(batch_size, 28, 28).detach().cpu().numpy()
    save_images(samples, save_path+'samples_{}.jpg'.format(iter))


def save_images(X, save_path, use_np=False):
    # [0, 1] -> [0,255]
    plt.ion()
    if not use_np:
        if isinstance(X.flatten()[0], np.floating):
            X = (255.99*X).astype('uint8')
    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1
    nh, nw = rows, int(n_samples/rows)
    if X.ndim == 2:
        s = int(np.sqrt(X.shape[1]))
        X = np.reshape(X, (X.shape[0], s, s))
    if X.ndim == 4:
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))
    for n, x in enumerate(X):
        j = int(n/nw)
        i = int(n%nw)
        img[j*h:j*h+h, i*w:i*w+w] = x

    plt.imshow(img, cmap='gray')
    plt.draw()
    plt.pause(0.001)
    if use_np:
        np.save(save_path, img)
    else:
        imsave(save_path, img)


def load_default_args():
    parser = argparse.ArgumentParser(description='default hyper-args')
    parser.add_argument('--z', default=128, type=int, help='latent space width')
    parser.add_argument('--ze', default=256, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--model', default='small2', type=str)
    parser.add_argument('--beta', default=1000, type=int)
    parser.add_argument('--use_x', default=False, type=bool)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--use_d', default=False, type=str)
    parser.add_argument('--boost', default=10, type=int)

    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='')
    parser.add_argument('--net', type=str, default='small2', metavar='N', help='')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='N', help='')
    parser.add_argument('--mdir', type=str, default='models/', metavar='N', help='')
    parser.add_argument('--scratch', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--ft', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--hyper', type=bool, default=False, metavar='N', help='')
    parser.add_argument('--task', type=str, default='train', metavar='N', help='')

    args = parser.parse_args([])
    return args

