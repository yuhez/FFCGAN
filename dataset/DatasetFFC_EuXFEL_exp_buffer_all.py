"""
    DataLoader for loading the FFC files from the Shimadzu camera
    Data format: 
        input: [20,128,250,400]
        output: [1, 1, 250,400]

    1. Pad the images from [250,400] to [256,512] for the training
    2. Load files from multiple .h5 files    
"""
import os
import re
import torch
import random
import h5py
import numpy as np
from extra_data import open_run

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
        self.camera_idx = config['data']['camera_idx']
        self.mode = mode.lower()
        if self.mode == 'train':
            self.run_list = config['data']['train_run_list'] 
            self.gt_file_list = self.get_gt_files(self.run_list)
        elif self.mode == 'test':
            self.run_list = config['data']['test_run_list']
            try:
                self.gt_file_list = self.get_gt_files(self.run_list)
            except:
                self.gt_file_list=[]
        else:
            raise ValueError("Please set the mode to 'train' or 'test'.")
        self.Shimadzu_num = config['data']['Shimadzu_num']
        self.h5_gt_pattern = config['data']['h5_gt_pattern']
        self.output_size = config['data']['pad_size']
        
        self.raw_camera_pattern = f'SPB_EHD_HPVX2_{self.camera_idx}/CAM/CAMERA:daqOutput'
        self.remove_first_frame = 1 # Remove the noise frames

        # Create buffers for quicker data loading
        self.raw_data = []
        self.gt_data = []
        for run in self.run_list:
            self.raw_data.append(self.load_extra_data(run)[:,self.remove_first_frame:])
        if  self.gt_file_list:    
            for gt_file in self.gt_file_list:
                # print(gt_file)
                # print(self.read_h5_all(gt_file, self.h5_gt_pattern).shape)
                self.gt_data.append(self.read_h5_all(gt_file, self.h5_gt_pattern)[:,self.remove_first_frame:])
            self.gt_data = np.concatenate(self.gt_data)
            if self.config['training']['preprocess']=='standardize':
                self.gt_data = data_reg(self.gt_data)

        self.raw_data = np.concatenate(self.raw_data)
        if self.config['training']['preprocess']=='standardize':
            print('standardizing input data')
            self.raw_data = data_reg(self.raw_data)
        # self.imgs_list = self.get_total_num_imgs_raw()
        # self.accum_list = np.cumsum(self.imgs_list)
        
    @staticmethod
    def read_h5_all(filename, pattern):
        """
            Load images from an h5 file
        """
        with h5py.File(filename, 'r') as f:
            data = f[pattern][:]
        return data
    
    def load_extra_data(self, run_number):
        # Load raw data using Extra-data
        run = open_run(proposal=self.config['data']['proposal_number'],run=run_number)
        data = run[self.raw_camera_pattern]['data.image.pixels'].ndarray()
        return data
        
    def get_gt_files(self, run_list):
        # Get all file paths to the flat-field corrected ones
        gt_file_list = []
        gt_path = self.config['data']['ffc_path']
        for run in run_list:
            if self.config['data']['TV_denoising']:
                run_name_format = f'r{run:04d}/TV_denoising'
                re_pattern = f'camera{self.camera_idx}_TVdenoised'
                tmp_pattern=''
            else:
                run_name_format = f'r{run:04d}'
                re_pattern = f'camera{self.camera_idx}_dffc_{run_name_format}'+ r'_flatr([0-9]+)_rank([0-9]+)_ds([0-9]+).h5'
                tmp_pattern = f'camera{self.camera_idx}_dffc_r{run}'+ r'_rank([0-9]+)_ds([0-9]+).h5'
            
            for file in os.listdir(os.path.join(gt_path,run_name_format)):
                if re.match(re_pattern, file) or re.match(tmp_pattern, file):
                    gt_file_list.append(os.path.join(gt_path,run_name_format,file))
        print('Loading GT files from', gt_file_list)
        return gt_file_list

#     def get_total_num_imgs_new(self):
#         """ Go through all datasets and find the total number of images """
#         imgs_list = []
#         for file in self.gt_file_list:
#             imgs_list.append(self.get_num_imgs(file, pattern=self.h5_gt_pattern))
#         return imgs_list # [234, 444,2444,]

    def get_num_imgs(self, filename, pattern):
        """ Get number of images in a single dataset """
        with h5py.File(filename, 'r') as f:
            assert f[pattern].ndim == 4
            num_time, frame_per_time, img_h, img_w = f[pattern].shape
            print(f'dataset shape for {filename} is:',num_time, frame_per_time, img_h, img_w)
        return num_time * frame_per_time
    
    def get_total_num_imgs_raw(self):
        """ Go through all RAW data and find the total number of images """
        imgs_list = []
        for data in self.raw_data:
            imgs_list.append(data.shape[0]*data.shape[1])
        return imgs_list

    def __len__(self):
        # return sum(self.imgs_list)
        return self.raw_data.shape[0]*self.raw_data.shape[1]


    # def find_file_from_index(self, index):
    #     if index< self.accum_list[0]:
    #             return 0
    #     for i in range(len(self.accum_list)):
    #         if index >= self.accum_list[i] and index < self.accum_list[i+1]:
    #             return i+1
    
    # def find_run_from_index(self, index):
    #     # Find the number of run from index
    #     return self.find_file_from_index(index)
    
    def get_img_online(self,index):
        # Find which run to load
        run_idx = self.find_run_from_index(index)
        if run_idx == 0:
            img_idx = index
        else:
            img_idx = index - self.accum_list[run_idx-1]
        img_idx_i = img_idx // self.Shimadzu_num
        img_idx_j = img_idx % self.Shimadzu_num
        # Load raw data using extra-data
        img = self.raw_data[run_idx][img_idx_i,img_idx_j]
        # print(run_idx)
        # print(len(self.raw_data), self.raw_data[0].shape)
        # print(len(self.gt_data), self.gt_data[0].shape)

#         img = self.load_extra_data(self.run_list[run_idx], self.raw_camera_pattern, img_idx_i,img_idx_j)
        # Load flat-field corrected images as ground truth from the h5 files prepared by PCA
        if self.gt_file_list:
#             gt_file_path = self.gt_file_list[run_idx]
#             gt = self.read_h5_index(gt_file_path, self.config['data']['h5_gt_pattern'],img_idx_i,img_idx_j)
            gt = self.gt_data[run_idx][img_idx_i,img_idx_j]
        else:
            gt = None
        return img, gt

    @staticmethod
    def normalize(x):
        x = (x - x.min())/(x.max()-x.min())
        return x

    def pre_process(self, img, gt):
        # Random horizontal and vertical flipping
        if self.mode == 'train':
            if random.random() > 0.5:
                img = np.flip(img,0)
                gt = np.flip(gt,0)
            if random.random() > 0.5:
                img = np.flip(img,1)
                gt = np.flip(gt,1)
        
        img, gt = self.pad_images(img, gt) # pad to pad_size
        if self.config['training']['preprocess'].lower()=='normalize':
            img = self.normalize(img)
            if self.gt_file_list:
                gt = self.normalize(gt)
        return img, gt

    def pad_images(self, image, mask):
        h,w = image.shape
        pad_h = self.output_size[0]-h
        pad_w = self.output_size[1]-w
        image = np.pad(image, ((pad_h//2, pad_h//2),(pad_w//2, pad_w//2)), 'constant')
        if self.gt_file_list:
            mask = np.pad(mask, ((pad_h//2, pad_h//2),(pad_w//2, pad_w//2)), 'constant')
            return image.copy(), mask.copy()
        else:
            return image.copy(), None


    def get_img_simple(self,index):
        img = self.raw_data.reshape(-1, *self.raw_data.shape[-2:])[index]
        if self.gt_file_list:
            gt = self.gt_data.reshape(-1, *self.gt_data.shape[-2:])[index]
        else:
            gt = None
        return img, gt

    def __getitem__(self, index):
        """ For a given index, find which file to load and which images"""
        img, gt = self.get_img_simple(index)
        img, gt = self.pre_process(img, gt)
        
        if self.gt_file_list:
            img = torch.from_numpy(img)
            gt = torch.from_numpy(gt)
            return (img, gt)
        else:
            img = torch.from_numpy(img)
            return img
            

def print_data_info(x):
    x = x.astype(np.float64)
    x = x.flatten()
    print(f'mean = {np.mean(x):3.3f}, min = {np.min(x):3.3f}, max = {np.max(x):3.3f}, median ={np.median(x):3.3f}, std={np.std(x):3.3f}')
    
def data_reg(images):
    """Regularization"""
    # images = np.array(images)
    images_mean = images.mean()
    images_std = images.std()
    images = (images - images_mean) / images_std
    # images_min = images.min()
    # images = images - images_min
    return images
    


