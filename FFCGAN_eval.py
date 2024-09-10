"""
GAN for flat-field correction at XFEL (Validation)
Save the output to h5 files
Author: Yuhe Zhang
Date: 2022-09-06
"""

import os
import torch
from torch import nn
from models import get_model, get_NLdnet
from utils.visualizer import *
from utils.others import *
import torchvision.transforms as transforms
from utils.config import load_config, save_config
from collections import OrderedDict
import itertools
from dataset.DatasetFFC_EuXFEL_exp_buffer_all import *
from FFCGAN import FFCGAN
from models.metrics import Metrics

class Evaluator(FFCGAN):
    """docstring for Evaluater"""
    def __init__(self, config):
        super(FFCGAN, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float

    def write_to_hdf(self, filename, pattern, data):
        with h5py.File(filename, 'w') as f:
            # data = f.create_group('FFC')
            f[pattern] = data

    def append_to_hdf(self, filename, layername, img):
        with h5py.File(filename, 'a') as f:
            # del f[layername]
            try: 
                f[layername] = img
            except:
                f[layername][...] = img

    def write_data(self, imgs):
        """ Save the output to h5 files. Append to the original file or save to new file?"""
        file_out = os.path.join(self.config['evaluation']['dir'], f"r{self.config['data']['test_run_list'][0]:04d}") #f
        self.create_dir_if_not_exist(file_out)
        # self.append_to_hdf(file_out, self.config['evaluation']['write_h5_pattern'], imgs)
        self.write_to_hdf(f"{file_out.split('.')[0]}/run{self.config['data']['test_run_list'][0]:04d}_trained_with_{self.config['evaluation']['expname']}.h5", self.config['evaluation']['write_h5_pattern'],imgs)

    def set_input(self, input_data):
        if len(input_data) == 2:
            self.images = input_data[0].to(self.device, dtype=self.dtype)
           
            self.gt = input_data[1].to(self.device, dtype=self.dtype)
        else:
            self.images = input_data.to(self.device, dtype=self.dtype)
            self.gt = None
            
        if self.images.ndim==3:
            self.images = self.images.unsqueeze(1)
    def crop_rec(self, dataset):
        """ Recover original image size from the padded results"""
        pad_h = (self.config['data']['pad_size'][0] - self.config['data']['img_size'][0])//2
        pad_w = (self.config['data']['pad_size'][1] - self.config['data']['img_size'][1])//2
        h,w = self.config['data']['img_size']
        return dataset[:,:,pad_h:pad_h+h,pad_w:pad_w+w]

    def run_network(self):
        """ Load the pretrained network and run it to get the predictions from the network"""
        netG = torch.nn.DataParallel(get_model(pretrained=False,init = False,num_out = 1)).to(device)
        self.load_trained_model(self.format_netname('G',f"{self.config['pretrain']['load_epoch']:03d}"), netG)
        test_loader = self.load_data()
        img_list=[]
        with torch.no_grad():
            for iters, test_data in enumerate(test_loader):
                self.set_input(test_data)
                fake = netG(self.images)
                fake = self.crop_rec(fake)
                if self.gt is not None:
                    real = self.gt.unsqueeze(1)
                    error = Metrics().cal_metrics(fake, self.crop_rec(real))
                    print(f"Step [{iters + 1}/{self.total_step}], MSE: {error['mse']:.5f}, SSIM: {error['ssim']:.4f}")
                else: 
                    print(f"Step [{iters + 1}/{self.total_step}]")
                img_list.append(fake.squeeze().detach().cpu().numpy())
        # img_list = np.array(img_list).reshape(-1,fake.shape[-2], fake.shape[-1]) # for reshape to 3D
        img_list = np.array(img_list)
        self.write_data(img_list)

    def load_data(self):
        print("start loading data....")
        self.test_dataset = DatasetFFC(self.config, mode='test')
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        self.total_step = len(self.test_loader)
        print("finish loading data")
        return self.test_loader

    @staticmethod
    def format_netname(net,ep):
        return F'net{net}_{ep}epoch.pt'

    def load_trained_model(self, model_load_name,net):
        path = F"{self.config['evaluation']['load_dir']}/{self.config['evaluation']['expname']}/save/{model_load_name}"
        # optimizer = torch.optim.Adam(net.parameters())
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()


if __name__ == '__main__':
    config = load_config('configs/eval.yaml','configs/default.yaml')
    MyEvaluator = Evaluator(config)
    MyEvaluator.run_network()
