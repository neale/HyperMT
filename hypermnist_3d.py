import sys
import torch
import pprint
import argparse
import numpy as np

from torch import optim
from torch.nn import functional as F

import ops
import utils
import netdef
import datagen


def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=128, type=int, help='latent space width')
    parser.add_argument('--ze', default=256, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--target', default='mednet', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--beta', default=100, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--use_x', default=False, type=bool)
    parser.add_argument('--pretrain_e', default=False, type=bool)
    parser.add_argument('--scratch', default=False, type=bool)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--model', default='small', type=str)

    args = parser.parse_args()
    return args


# hard code the two layer net
def _train_clf(args, Z, data, target):
    """ calc classifier loss on target architecture """
    data, target = data.cuda(), target.cuda()
    out = F.conv2d(data, Z[0], stride=1)
    out = F.leaky_relu(out)
    out = F.max_pool2d(out, 2, 2)
    out = F.conv2d(out, Z[1], stride=1)
    out = F.leaky_relu(out)
    out = F.max_pool2d(out, 2, 2)
    out = out.view(-1, 512)
    out = F.linear(out, Z[2])
    loss = F.cross_entropy(out, target)
    pred = out.data.max(1, keepdim=True)[1]
    correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
    return (correct, loss)

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


def z_loss(args, real, fake):
    zero = torch.zeros_like(fake)
    one = torch.ones_like(real)
    d_fake = F.mse(fake, one)
    d_real = F.binary_cross_entropy_with_logits(real, zero)
    d_real_trick = F.binary_cross_entropy_with_logits(real, one)
    loss_z = 10 * (d_fake + d_real)
    return loss_z, d_real_trick


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

    optimE = optim.Adam(netE.parameters(), lr=5e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW1 = optim.Adam(W1.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW2 = optim.Adam(W2.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW3 = optim.Adam(W3.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW4 = optim.Adam(W3.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimW5 = optim.Adam(W3.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    
    best_test_acc, best_test_loss = 0., np.inf
    args.best_loss, args.best_acc = best_test_loss, best_test_acc

    mnist_train, mnist_test = datagen.load_mnist(args)
    x_dist = utils.create_d(args.ze)
    z_dist = utils.create_d(args.z)
    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()
    print ("==> pretraining encoder")
    j = 0
    final = 100.
    e_batch_size = 1000
    if args.pretrain_e:
        for j in range(100):
            x = utils.sample_d(x_dist, e_batch_size)
            z = utils.sample_d(z_dist, e_batch_size)
            codes = netE(x)
            for i, code in enumerate(codes):
                code = code.view(e_batch_size, args.z)
                mean_loss, cov_loss = ops.pretrain_loss(code, z)
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
        for batch_idx, (data, target) in enumerate(mnist_train):
            netE.zero_grad(); W1.zero_grad(); W2.zero_grad()
            W3.zero_grad(); W4.zero_grad(); W5.zero_grad()
            z = utils.sample_d(x_dist, args.batch_size)
            codes = netE(z)
            #ops.free_params([netD]); ops.frozen_params([netE, W1, W2, W3])
            for code in codes:
                noise = utils.sample_z_like((args.batch_size, args.z))
                d_real = netD(noise)
                d_fake = netD(code)
                d_real_loss = torch.log((1-d_real).mean())
                d_fake_loss = torch.log(d_fake.mean())
                d_real_loss.backward(torch.tensor(-1, dtype=torch.float).cuda(),retain_graph=True)
                d_fake_loss.backward(torch.tensor(-1, dtype=torch.float).cuda(),retain_graph=True)
                d_loss = d_real_loss + d_fake_loss
            optimD.step()
            #ops.frozen_params([netD])
            #ops.free_params([netE, W1, W2, W3])
            netD.zero_grad()
            z = utils.sample_d(x_dist, args.batch_size)
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
                scaled_loss.backward(retain_graph=True)
                d_loss.backward(torch.tensor(-1, dtype=torch.float).cuda(),retain_graph=True)
            optimE.step(); optimW1.step()
            optimW2.step(); optimW3.step()
            optimW4.step(); optimW5.step()
            loss = loss.item()
                
            if batch_idx % 50 == 0:
                acc = correct
                print ('**************************************')
                print ('{} MNIST Test, beta: {}'.format(args.model, args.beta))
                print ('Acc: {}, Loss: {}'.format(acc, loss))
                print ('D loss: {}'.format(d_loss))
                print ('best test loss: {}'.format(args.best_loss))
                print ('best test acc: {}'.format(args.best_acc))
                print ('**************************************')
            
            if batch_idx > 1 and batch_idx % 199 == 0:
                test_acc = 0.
                test_loss = 0.
                for i, (data, y) in enumerate(mnist_test):
                    z = utils.sample_d(x_dist, args.batch_size)
                    codes = netE(z)
                    l1 = W1(codes[0])
                    l2 = W2(codes[1])
                    l3 = W3(codes[2])
                    l4 = W4(codes[3])
                    l5 = W5(codes[4])
                    for (g1, g2, g3, g4, g5) in zip(l1, l2, l3, l4, l5):
                        correct, loss = train_clf(args, [g1, g2, g3, g4, g5], data, y)
                        test_acc += correct.item()
                        test_loss += loss.item()
                test_loss /= len(mnist_test.dataset) * args.batch_size
                test_acc /= len(mnist_test.dataset) * args.batch_size
        
                print ('Test Accuracy: {}, Test Loss: {}'.format(test_acc, test_loss))
                if test_loss < best_test_loss or test_acc > best_test_acc:
                    print ('==> new best stats, saving')
                    #utils.save_clf(args, z_test, test_acc)
                    if test_acc > .85:
                        utils.save_hypernet_cifar(args, [netE, W1, W2, W3, W4, W5], test_acc)
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        args.best_loss = test_loss
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        args.best_acc = test_acc


if __name__ == '__main__':

    args = load_args()
    import models.models_cifar_small as models
    modeldef = netdef.nets()[args.target]
    pprint.pprint (modeldef)
    # log some of the netstat quantities so we don't subscript everywhere
    args.stat = modeldef
    args.shapes = modeldef['shapes']
    train(args)
