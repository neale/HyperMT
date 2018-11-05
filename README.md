# HyperMT
Newest addition to the HyperGAN family. Generate multi-task networks

First up is an AutoEncoder generated with HyperGAN

## Autoencoder
![plot](plots/samples_1194.jpg)
![plot2](plots/samples_1393.jpg)

## Multiple Datasets
Using dataset-conditional variables we can generate networks which achieve high classification scores on multiple datasets. A single latent space in this instance is enough to contain all the mappings we need

There is a small accuracy penalty, but nothing too severe. 

* MNIST - 99% accuracy
* CIFAR-10 75% accuracy
