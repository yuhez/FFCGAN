"""
    DataLoader for loading the FFC files from the Shimadzu camera
    Data format: 
        input: [20,128,250,400]
        output: [1, 1, 250,400]

    1. Pad the images from [250,400] to [256,512] for the training
    2. Load files from multiple .h5 files    
"""
import os
import torch
import random
import h5py
import numpy as np

def read_h5(filename, pattern):
    """
        Load images from an h5 file
    """
    with h5py.File(filename, 'r') as f:
        data = f[pattern][:]
    return data

class DatasetFFC(torch.utils.data.Dataset):
    """docstring for ClassName"""
    def __init__(self, config, mode='train'):
        super().__init__()
        self.config = config
        self.file_path = config['data']['path']
        self.mode = mode.lower()
        if self.mode == 'train':
            self.file_list =  config['data']['train'] # ['FFC_Venturi_run_60.h5','FFC_Venturi_run_61.h5']
        elif self.mode == 'test':
            self.file_list =  config['data']['test'] # ['FFC_Venturi_run_60.h5','FFC_Venturi_run_61.h5']
        else:
            raise ValueError("Please set the mode to 'train' or 'test'.")
        self.Shimadzu_num = config['data']['Shimadzu_num']
        self.h5_data_pattern = config['data']['h5_pattern'][0]
        self.h5_gt_pattern = config['data']['h5_pattern'][1]
        self.output_size = config['data']['pad_size']

        self.imgs_list = self.get_total_num_imgs()
        self.accum_list = np.cumsum(self.imgs_list)

    def get_total_num_imgs(self):
        """ Go through all datasets and find the total number of images """
        imgs_list = []
        for file in self.file_list:
            file_in = os.path.join(self.file_path,file)
            imgs_list.append(self.get_num_imgs(file_in, pattern=self.h5_gt_pattern))
        return imgs_list # [234, 444,2444,]

    def get_num_imgs(self, filename, pattern):
        """ Get number of images in a single dataset """
        with h5py.File(filename, 'r') as f:
            assert f[pattern].ndim == 4
            num_time, frame_per_time, img_h, img_w = f[pattern].shape
            print(f'dataset shape for {filename} is:',num_time, frame_per_time, img_h, img_w)
        return num_time * frame_per_time

    def __len__(self):
        return sum(self.imgs_list)

    def find_file_from_index(self, index):
        if index< self.accum_list[0]:
                return 0
        for i in range(len(self.accum_list)):
            if index >= self.accum_list[i] and index < self.accum_list[i+1]:
                return i+1

    @staticmethod
    def read_h5_index(filename, pattern,i,j):
        """
            Load images from an h5 file
        """
        with h5py.File(filename, 'r') as f:
            data = f[pattern][i,j]
        return data


    def get_img(self, index):
        # print(index)
        file_idx = self.find_file_from_index(index) # find which file to load
        # print(self.file_list[file_idx])
        if file_idx == 0:
            img_idx = index
        else:
            img_idx = index - self.accum_list[file_idx-1]
        img_idx_i = img_idx // self.Shimadzu_num
        img_idx_j = img_idx % self.Shimadzu_num
        file = os.path.join(self.file_path,self.file_list[file_idx])
        img = self.read_h5_index(file, self.h5_data_pattern,img_idx_i,img_idx_j)
        gt = self.read_h5_index(file, self.h5_gt_pattern,img_idx_i,img_idx_j)
        return img, gt

    @staticmethod
    def normalize(x):
        x = (x - x.min())/(x.max()-x.min())
        return x

    def pre_process(self, img, gt):
        # Random horizontal and vertical flipping
        # if self.mode == 'train':
        #     if random.random() > 0.5:
        #         img = np.flip(img,0)
        #         gt = np.flip(gt,0)
        #     if random.random() > 0.5:
        #         img = np.flip(img,1)
        #         gt = np.flip(gt,1)
        
        img, gt = self.pad_images(img, gt) # pad to pad_size
        img = self.normalize(img)
        gt = self.normalize(gt)

        return img, gt

    def pad_images(self, image, mask):
        h,w = image.shape
        pad_h = self.output_size[0]-h
        pad_w = self.output_size[1]-w
        image = np.pad(image, ((pad_h//2, pad_h//2),(pad_w//2, pad_w//2)), 'constant')
        mask = np.pad(mask, ((pad_h//2, pad_h//2),(pad_w//2, pad_w//2)), 'constant')
        return image.copy(), mask.copy()

    def __getitem__(self, index):
        """ For a given index, find which file to load and which images"""
        img, gt = self.get_img(index)
        img, gt = self.pre_process(img, gt)
        # print('='*10,'images','='*10)
        # print_data_info(img)
        # print('='*10,'ground truth','='*10)
        # print_data_info(gt)
        img = torch.from_numpy(img)
        gt = torch.from_numpy(gt)
        return (img, gt)

def print_data_info(x):
    x = x.astype(np.float64)
    x = x.flatten()
    print(f'mean = {np.mean(x):3.3f}, min = {np.min(x):3.3f}, max = {np.max(x):3.3f}, median ={np.median(x):3.3f}, std={np.std(x):3.3f}')
    
def data_reg(images):
    """Regularization"""
    images_mean = images.mean()
    images_std = images.std()
    images = (images - images_mean) / images_std
    images_min = images.min()
    images = images - images_min
    return images
    