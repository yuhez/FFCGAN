"""
GAN for flat-field correction at XFEL
Author: Yuhe Zhang
Date: 2022-08-31
"""

import torch
from torch import nn
from models import get_model, get_NLdnet
# from utils.visualizer import *
from utils.others import *
import torchvision.transforms as transforms
from utils.config import load_config, save_config
from collections import OrderedDict
import itertools
from dataset.DatasetFFC_EuXFEL_exp_buffer_all import *
from models.metrics import Metrics
from utils.visualizer import Visualizer
import matplotlib.pyplot as plt

# from karabo_bridge import Client
# krb = Client("tcp://10.253.0.52:54333")
# krb.next()
# torch.manual_seed(1)
# np.random.seed(18)

class FFCGAN(object):
    """
        FFC GAN Trainer
    """
    def __init__(self, config):
        super(FFCGAN,self).__init__()
        self.config = config
        self.pretrain = self.config['pretrain']['use_pretrain']
        self.init_netG = False
        # self.init_netG = True if not self.pretrain else False
        self.loss_names = ['G', 'D', 'fsc', 'mse', 'total']
        self.visual_names = ['images', 'fake','real'] 
        self.predefine()
        self.dtype = torch.float
        self.criterion = nn.MSELoss()

    def set_input(self, input_data):
        # self.images = input_data[0].to(self.device, dtype=self.dtype)
        # if self.images.ndim==3:
        #     self.images = self.images.unsqueeze(1)
        # self.gt = input_data[1].to(self.device, dtype=self.dtype)
        if len(input_data) == 2:
            self.images = input_data[0].to(self.device, dtype=self.dtype)
           
            self.gt = input_data[1].to(self.device, dtype=self.dtype)
        else:
            self.images = input_data.to(self.device, dtype=self.dtype)
            self.gt = None

        if self.images.ndim==3:
            self.images = self.images.unsqueeze(1)

    def init_models(self):
        self.init_networks()
        self.load_data_new()

    def print_val(self, error, logfile, epoch, iters):
        message = "-> Epoch [{}/{}], Step [{}/{}]".format(
            epoch + 1, self.config['training']['num_epochs'], iters + 1, self.total_step_val
        )
        message += f", MSE: {error['mse']:.5f}, SSIM: {error['ssim']:.4f}"
        print(message)
        with open(
            logfile,
            "a",
            encoding="utf-8",
        ) as f:
            print(message, file=f)

    def quick_eval(self, **kwargs):
        self.fake = self.netG(self.images)
        if self.gt is not None:
            self.real = self.gt.unsqueeze(1)
            error = Metrics().cal_metrics(self.fake,self.real)
            if error['ssim']>= self.ssim_best:
                self.ssim_best = error['ssim']
                save_model('best_netG', F"{Trainer.wdir}/save", f'{epoch+1:03d}', Trainer.netG, Trainer.optimizer_G, Trainer.G_loss)
            self.print_val(error=error, logfile=self.save_val, **kwargs)

    def load_data(self):
        self.train_loader, self.test_loader = self.load_data_2channel(self.config['training']['batch_size'], self.config['data']['path'])
        self.total_step = len(self.train_loader)

    def load_data_new(self):
        print("start loading data....")
        self.train_dataset = DatasetFFC(self.config)
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        self.total_step = len(self.train_loader)
        self.test_dataset = DatasetFFC(self.config, mode='test')
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        self.total_step_val = len(self.test_loader)

        print("finish loading data")

    def init_networks(self):
        self.netG = torch.nn.DataParallel(get_model(pretrained=True,init = self.init_netG,num_out = 1)).to(self.device)
        self.netD = torch.nn.DataParallel(get_NLdnet(1)).to(self.device)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.config['training']['lr_d'], betas=(self.config['training']['beta1'], 0.999))
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.config['training']['lr_g'], betas=(self.config['training']['beta1'], 0.999))
        if self.pretrain:
            load_pretrain_model = self.format_netname('G',f"{self.config['pretrain']['load_epoch']:03d}",self.config['pretrain']['load_run'])
            self.load_trained_model(load_pretrain_model,self.netG)

    def predefine(self, wdir='./results'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
        self.wdir = os.path.join(config['training']['dir'], config['expname'])
        self.save_log = F"{self.wdir}/log.txt"
        self.save_val = F"{self.wdir}/val.txt"
        self.create_dir_if_not_exist(self.wdir)

    def init_training(self):
        self.lr_g =  self.config['training']['lr_g']
        self.lr_decay_factor = self.config['training']['lr_decay_factor'] 
        self.lr_decay_epochs = self.config['training']['lr_decay_epochs']

    def adjust_learning_rate(self, optimizer, epoch, lr_start):
        """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
        lr = lr_start * (
            self.config['training']['lr_decay_factor'] ** (epoch // self.config['training']['lr_decay_epochs'])
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def load_trained_model(self,model_load_name,net):
        path = F"{self.config['training']['dir']}/{self.config['pretrain']['load_run']}/save/{model_load_name}"
        # optimizer = torch.optim.Adam(net.parameters())
        print('Loading pretrained model from ', path)
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()

    @staticmethod
    def format_netname(net,ep,foldername):
        loadname = F'net{net}_{ep}epoch.pt'
        return loadname

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def plot_cycle(self,img_idx, layer, save_name, folder='train', cmap='bone'):
        img_list = ['Input', 'Output', 'Input','Ground truth']
        data_list = [self.images, self.fake,self.images, self.real]

        fig, axs = plt.subplots(2, int(len(img_list) / 2), figsize=(20, 10), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=0.0001, wspace=0.0001)
        axs = axs.ravel()
        for i in range(len(img_list)):
            im = axs[i].imshow(data_list[i][img_idx, layer, :, :].detach().cpu(),cmap=cmap)
            axs[i].axis("off")
            axs[i].set_title(img_list[i], fontsize=48)
        # cbar_ax = fig.add_axes([0.92, 0.25, 0.05, 0.5])
        # fig.colorbar(im, cax=cbar_ax)
        save_path = F"{self.wdir}/{folder}"
        self.create_dir_if_not_exist(save_path)
        plt.tight_layout()
        plt.savefig(save_path + F"/{save_name}.png")
        plt.cla()
        plt.close()

    def get_losses(self, loss_list):
        errors_list = OrderedDict()
        for name in loss_list:
            if isinstance(name, str):
                errors_list[name] = float(getattr(self, name + "_loss"))
        return errors_list

    def print_losses(self, logfile, epoch, iters, losses):
        message = "Epoch [{}/{}], Step [{}/{}]".format(
            epoch + 1, self.config['training']['num_epochs'], iters + 1, self.total_step
        )
        for name, loss in losses.items():
            message += ", {:s}: {:.3f}".format(name, loss)
        print(message)
        with open(
            logfile,
            "a",
            encoding="utf-8",
        ) as f:
            print(message, file=f)

    def get_current_losses(self):
        return self.get_losses(self.loss_names)

    def print_current_losses(self, **kwargs):
        self.print_losses(self.save_log, **kwargs)

    def plot_loss(self, save_name=0, ylim_list=[-1, 10]):
        plt.figure(figsize=(20, 10))
        plt.rcParams.update({'font.size': 22})
        plt.rcParams.update({'lines.linewidth': 3})
        plt.title("Loss During Training")

        plt.plot(self.total_loss.detach().cpu().numpy(), label="Total loss")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        if ylim_list != 0:
            plt.ylim(ylim_list)

        plt.legend(prop={"size": 20})
        save_path = F"{self.wdir}/loss"
        self.create_dir_if_not_exist(save_path)
        plt.savefig(save_path + F"/{save_name}.png")
        plt.cla()
        plt.close()

    @staticmethod
    def create_dir_if_not_exist(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def check_folder_exist(self, folder_name):
        path = F"{self.wdir}/{folder_name}"
        if os.path.exists(path):
            decision = input('This folder already exists. Continue training will overwrite the data. Proceed(y/n)?')
            if decision != 'y':
                exit()
            else:
                print('Warning: Overwriting folder: {}'.format(folder_name))

    def update_parameters(self, epoch):
            self.adjust_learning_rate(self.optimizer_G, epoch, lr_start=self.config['training']['lr_g'])
            self.adjust_learning_rate(self.optimizer_D, epoch, lr_start=self.config['training']['lr_d'])

    def train(self):

        self.optimizer_G.zero_grad()
        self.real = self.gt.unsqueeze(1)
        self.fake = self.netG(self.images)

        self.optimizer_D.zero_grad()
        self.DA_realB, self.DA_fakeB,self.D_loss = backward_D_basic(self.netD, self.real, self.fake)
        self.optimizer_D.step()

        self.G_loss = GANLoss(self.netD(self.fake),True) * self.config['training']['lambda_G']
        if self.config['training']['lambda_fsc']>0:
            self.fsc_loss = self.config['training']['lambda_fsc'] * FSC_MSE_loss(self.real.squeeze(),self.fake.squeeze(),self.images.shape[0])
        else:
            self.fsc_loss = torch.tensor(0)
        self.mse_loss = self.config['training']['lambda_mse'] * self.criterion(self.fake,self.real)
        self.total_loss = self.mse_loss + self.G_loss + self.fsc_loss
        self.total_loss.backward()
        if self.config['training']['clip_max'] > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.config['training']['clip_max'])
        self.optimizer_G.step()


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

    def write_data(self, imgs, epoch):
        """ Save the output to h5 files. Append to the original file or save to new file?"""
        path_out = os.path.join(self.config['evaluation']['dir'], f"r{self.config['data']['test_run_list'][0]:04d}") #f
        self.create_dir_if_not_exist(path_out)
        file_out = f"{path_out.split('.')[0]}/run{self.config['data']['test_run_list'][0]:04d}_trained_with_{self.config['expname']}.h5"
        try:
            self.append_to_hdf(file_out,f'{epoch:03d}epoch',imgs)
        except:
            self.write_to_hdf(file_out, f'{epoch:03d}epoch',imgs)

    def crop_rec(self, dataset):
        """ Recover original image size from the padded results"""
        pad_h = (self.config['data']['pad_size'][0] - self.config['data']['img_size'][0])//2
        pad_w = (self.config['data']['pad_size'][1] - self.config['data']['img_size'][1])//2
        h,w = self.config['data']['img_size']
        return dataset[:,:,pad_h:pad_h+h,pad_w:pad_w+w]

if __name__ == '__main__':

    config = load_config('configs/FFC_exp.yaml','configs/default.yaml')
    Trainer = FFCGAN(config)
    print('device:{}'.format(device))

    save_config(os.path.join(Trainer.wdir, 'training_config.yaml'), config)

    folder_name = config['expname']
    Trainer.check_folder_exist(folder_name)
    log_note = 'Modified 20210120; pairedUnetADV: supervised learning'

    Trainer.init_models()
    if config['visualization']['display_id'] > 0:
        Viz = Visualizer(config)
    # Train the model
    running_loss = 0.0
    Trainer.ssim_best = 0.0
    for epoch in range(config['training']['num_epochs']):
        time_start = time.time()
        if config['visualization']['display_id'] > 0:
            Viz.reset()
        Trainer.update_parameters(epoch)
        for i, train_data in enumerate(Trainer.train_loader):

            Trainer.set_input(train_data)
            Trainer.train()

            if i == 0 and epoch == 0:
                Trainer.plot_cycle(0, 0, 'startpoint_ie0'.format(epoch + 1))
                Trainer.plot_cycle(1, 0, 'startpoint_ie1'.format(epoch + 1))
                Trainer.plot_cycle(2, 0, 'startpoint_ie2'.format(epoch + 1))

            if i % config['training']['print_loss_freq_iter'] == config['training']['print_loss_freq_iter'] - 1:
                losses = Trainer.get_current_losses()
                Trainer.print_current_losses(epoch=epoch, iters=i, losses=losses)
                if config['visualization']['display_id'] > 0:
                    Viz.plot_current_losses(epoch, float(i) / Trainer.total_step, losses)
        if (
            epoch == 0
            or epoch % config['training']['save_plot_freq_epoch'] == config['training']['save_plot_freq_epoch'] - 1
        ):
            Trainer.plot_cycle(0, 0, '{:03d}epoch_{:04d}step'.format(epoch + 1, i + 1))
            save_result = Trainer.total_step % config['training']['save_plot_freq_epoch'] == 0
            if config['visualization']['display_id'] > 0:
                Viz.display_current_results(Trainer.get_current_visuals(), epoch, save_result)

            # Evaluation
            with torch.no_grad():
                val_list = []
                for i, test_data in enumerate(Trainer.test_loader):
                    # if i<=2:
                    Trainer.set_input(test_data)
                    Trainer.quick_eval(epoch=epoch, iters=i)
                    Trainer.plot_cycle(0, 0, '{:03d}epoch_{:04d}step'.format(epoch + 1, i + 1),folder='test/0')
                    Trainer.plot_cycle(-1, 0, '{:03d}epoch_{:04d}step'.format(epoch + 1, i + 1),folder='test/-1')
                    fake = Trainer.crop_rec(Trainer.fake)
                    val_list.append(fake.squeeze().detach().cpu().numpy())
                val_list=np.array(val_list)
                # print(val_list.dtype)

                Trainer.write_data(np.array(val_list).astype(np.float32)[:,::config['evaluation']['save_every_frames']], epoch + 1)

        if epoch % config['training']['save_model_freq_epoch'] == config['training']['save_model_freq_epoch'] - 1:
            save_model('netG', F"{Trainer.wdir}/save", f'{epoch + 1:03d}', Trainer.netG, Trainer.optimizer_G, Trainer.G_loss)

        time_end = time.time()
        train_time = time_end - time_start

        print(
            F'Training time for epoch {epoch}: {train_time // 3600} h {(train_time % 3600) // 60} min {(train_time % 3600) % 60} s')

    print('Finished Training')

