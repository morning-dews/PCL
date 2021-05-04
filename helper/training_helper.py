import helper.cifar10_dataset as cifar10_dataset
import helper.cifar100_dataset as cifar100_dataset
import helper.imagenet_dataset as ImagenetDataset
import helper.mnist_dataset as mnist_dataset
import helper.emnist_dataset as emnist_dataset
import helper.twentynews as twentynews
import helper.DBPedia as DBPedia
# import network

__author__ = 'garrett_local'


def load_dataset(dataset_name):

    if dataset_name == 'mnist':
        return mnist_dataset.MnistDataset
    if dataset_name == 'cifar10':
        return cifar10_dataset.Cifar10Dataset
    if dataset_name == 'cifar100':
        return cifar100_dataset.Cifar100Dataset
    if dataset_name == 'imagenet':
        return ImagenetDataset.ImagenetIterator
    if dataset_name == 'emnist':
        return emnist_dataset.EMnistDataset
    if dataset_name == 'twentynews':
        return twentynews.twentynews
    if dataset_name == 'dbpedia':
        return DBPedia.DBPedia


# def load_network(cfg):
#     network_name = cfg['network']['network_name']
#     if network_name == 'MLP':
#         return network.MultilayerPerceptron
#     if network_name == 'CNN':
#         return network.ConvolutionalNeuralNetwork
