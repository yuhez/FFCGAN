import torch
import numpy as np
from skimage.metrics import structural_similarity


class Metrics(object):
    def __init__(self):
        super(Metrics,self).__init__()

    @staticmethod
    def mse(predict, ground_truth):
        return np.mean((predict - ground_truth) ** 2)

    def l2(self, predict, ground_truth):
        return np.sqrt(self.mse(predict, ground_truth))

    def psnr(self, predict, ground_truth):
        mean_squared_error = self.mse(predict, ground_truth)
        return -10.0 * np.log(mean_squared_error) / np.log(10.0)

    @staticmethod
    def ssim(predict, ground_truth):
        return structural_similarity(predict, ground_truth)

    def dssim(self, predict, ground_truth):
        return (1 - self.ssim(predict, ground_truth))/2 

    def cal_metrics(self, img, gt):
        error = {}
        img = img.squeeze().detach().cpu().numpy()
        gt = gt.squeeze().detach().cpu().numpy()
        mse = self.mse(img, gt)
        error['mse'] = mse
        ssim = structural_similarity(img, gt, multichannel=True)
        error['ssim'] = ssim
#         print(f'MSE: {mse:.5f}, SSIM: {ssim:.4f}')
        return error