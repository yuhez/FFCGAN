import torch
from models.networks import PRNet,SRNet
from models.discriminator import Discriminator,NLayerDiscriminator
from models.initialization import init_weights, weights_init


def get_model(pretrained,init = False, type= 'normal',num_out = 1,num_in = 1):
    """FOR PROP-CYCLE GAN"""
    model = PRNet(pretrained = pretrained,num_out = num_out, num_in = num_in)
    model.eval()
    if init is True:
        init_weights(model, type, init_gain=0.02)
    return model

def get_dnet():
    dnet = Discriminator()
    # dnet = NLayerDiscriminator(1)
    dnet.eval()
    dnet.apply(weights_init)
    return dnet

def get_NLdnet(num_input=1,n_layers=3):
    # dnet = Discriminator()
    dnet = NLayerDiscriminator(num_input,n_layers=n_layers)
    dnet.eval()
    # init_weights(dnet, 'normal', init_gain=0.02)
    dnet.apply(weights_init)
    return dnet

