import sys
import torch
import pprint
import argparse
import numpy as np

from torch.nn import functional as F

import ops
import utils
import netdef
import datagen
from ae import Encoder as sampleE
from ae import Decoder as sampleD

def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=128, type=int, help='latent space width')
    parser.add_argument('--ze', default=256, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--target', default='small2', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--beta', default=100, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--use_x', default=False, type=bool)
    parser.add_argument('--pretrain_e', default=False, type=bool)
    parser.add_argument('--scratch', default=False, type=bool)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--use_d', default=False, type=str)
    parser.add_argument('--model', default='small', type=str)

    args = parser.parse_args()
    return args


# hard code the two layer net
def train_clf(args, Z, data, target):
    """ Encoder """
    data = data.cuda()
    out = F.conv2d(data, Z[0], stride=2, padding=2)
    out = F.dropout(out, p=0.3)
    out = F.relu(out)
    out = F.conv2d(out, Z[1], stride=2, padding=2)
    out = F.dropout(out, p=0.3)
    out = F.relu(out)
    out = F.conv2d(out, Z[2], stride=2, padding=2)
    out = F.dropout(out, p=0.3)
    out = F.relu(out)
    out = out.view(-1, 4*4*4*16)
    out = F.linear(out, Z[3])
    e_out = out.view(-1, 16)
    """ Decoder """
    out = F.linear(e_out, Z[4])
    out = out.view(-1, 4*16, 4, 4)
    out = F.conv_transpose2d(out, Z[5])
    out = F.relu(out)
    out = out[:, :, :7, :7]
    out = F.conv_transpose2d(out, Z[6])
    out = F.relu(out)
    d_out = F.conv_transpose2d(out, Z[7], stride=2)
    
    out = torch.sigmoid(d_out) 
    loss = F.mse_loss(out, data)
    return loss, out


def z_loss(args, real, fake):
    zero = torch.zeros_like(fake)
    one = torch.ones_like(real)
    d_fake = F.mse(fake, one)
    d_real = F.binary_cross_entropy_with_logits(real, zero)
    d_real_trick = F.binary_cross_entropy_with_logits(real, one)
    loss_z = 10 * (d_fake + d_real)
    return loss_z, d_real_trick


def train(args):
    from torch import optim
    torch.manual_seed(8734)
    netE = models.Encoderz(args).cuda()
    netD = models.DiscriminatorZ(args).cuda()
    E1 = models.GeneratorE1(args).cuda()
    E2 = models.GeneratorE2(args).cuda()
    E3 = models.GeneratorE3(args).cuda()
    E4 = models.GeneratorE4(args).cuda()
    D1 = models.GeneratorD1(args).cuda()
    D2 = models.GeneratorD2(args).cuda()
    D3 = models.GeneratorD3(args).cuda()
    D4 = models.GeneratorD4(args).cuda()
    print (netE, netD)
    print (E1, E2, E3, E4, D1, D2, D3, D4)

    optimE = optim.Adam(netE.parameters(), lr=5e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    
    Eoptim = [
        optim.Adam(E1.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4),
        optim.Adam(E2.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4),
        optim.Adam(E3.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4),
        optim.Adam(E4.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    ]
    Doptim = [
        optim.Adam(D1.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4),
        optim.Adam(D2.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4),
        optim.Adam(D3.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4),
        optim.Adam(D4.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    ]

    Enets = [E1, E2, E3, E4]
    Dnets = [D1, D2, D3, D4]

    best_test_loss = np.inf
    args.best_loss = best_test_loss

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
            netE.zero_grad()
            for optim in Eoptim:
                optim.zero_grad()
            for optim in Doptim:
                optim.zero_grad()
            z = utils.sample_d(x_dist, args.batch_size)
            codes = netE(z)
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
            netD.zero_grad()
            z = utils.sample_d(x_dist, args.batch_size)
            codes = netE(z)
            Eweights, Dweights = [], []
            i = 0
            for net in Enets:
                Eweights.append(net(codes[i]))
                i += 1
            for net in Dnets:
                Dweights.append(net(codes[i]))
                i += 1
            d_real = []
            for code in codes:
                d = netD(code)
                d_real.append(d)
                
            netD.zero_grad()
            d_loss = torch.stack(d_real).log().mean() * 10.

            for layers in zip(*(Eweights+Dweights)):
                loss, _ = train_clf(args, layers, data, target)
                scaled_loss = args.beta * loss
                scaled_loss.backward(retain_graph=True)
                d_loss.backward(torch.tensor(-1, dtype=torch.float).cuda(),retain_graph=True)
            optimE.step(); 
            for optim in Eoptim:
                optim.step()
            for optim in Doptim:
                optim.step()
            loss = loss.item()
                
            if batch_idx % 50 == 0:
                print ('**************************************')
                print ('AE MNIST Test, beta: {}'.format(args.beta))
                print ('MSE Loss: {}'.format(loss))
                print ('D loss: {}'.format(d_loss))
                print ('best test loss: {}'.format(args.best_loss))
                print ('**************************************')
            
            if batch_idx > 1 and batch_idx % 49 == 0:
                test_acc = 0.
                test_loss = 0.
                for i, (data, y) in enumerate(mnist_test):
                    z = utils.sample_d(x_dist, args.batch_size)
                    codes = netE(z)
                    Eweights, Dweights = [], []
                    i = 0
                    for net in Enets:
                        Eweights.append(net(codes[i]))
                        i += 1
                    for net in Dnets:
                        Dweights.append(net(codes[i]))
                        i += 1
                    for layers in zip(*(Eweights+Dweights)):
                        loss, out = train_clf(args, layers, data, y)
                        test_loss += loss.item()
                test_loss /= len(mnist_test.dataset) * args.batch_size
                print ('Test Loss: {}'.format(test_loss))
                if test_loss < best_test_loss:
                    print ('==> new best stats, saving')
                    #utils.save_clf(args, z_test, test_acc)
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        args.best_loss = test_loss
                archE = sampleE(args).cuda()
                archD = sampleD(args).cuda()
                eweight = list(zip(*Eweights))[0]
                dweight = list(zip(*Dweights))[0]
                modelE = utils.weights_to_clf(eweight, archE, args.statE['layer_names'])
                modelD = utils.weights_to_clf(dweight, archD, args.statD['layer_names'])
                utils.generate_image(args, batch_idx, modelE, modelD, data.cuda())


if __name__ == '__main__':
    args = load_args()
    import models.models_ae as models
    modeldef1 = netdef.nets()['aeE']
    modeldef2 = netdef.nets()['aeD']
    pprint.pprint(modeldef1)
    pprint.pprint(modeldef2)
    # log some of the netstat quantities so we don't subscript everywhere
    args.statE = modeldef1
    args.statD = modeldef2
    args.shapesE = modeldef1['shapes']
    args.shapesD = modeldef2['shapes']
    train(args)
