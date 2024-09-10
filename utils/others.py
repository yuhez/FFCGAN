import torch
from torch import nn
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def backward_D_new(netD, real, fake):
    """Calculate GAN loss for the discriminator
    Parameters:
        netD (network)      -- the discriminator D
        real (tensor array) -- real images
        fake (tensor array) -- images generated by a generator
    Return the discriminator loss.
    We also call loss_D.backward() to calculate the gradients.
    """
    # Real
    pred_real = netD(real)

    loss_D_real = GANLoss(pred_real, True)
    D_real = pred_real.view(-1).mean().item()
    loss_D_real.backward()
    # Fake
    pred_fake = netD(fake.detach())
    loss_D_fake = GANLoss(pred_fake, False)
    D_fake = pred_fake.view(-1).mean().item()
    loss_D_fake.backward()
    loss_D = (loss_D_real + loss_D_fake)
    # loss_D.backward()
    # return nn.Sigmoid(D_real), nn.Sigmoid(D_fake), loss_D
    return D_real, D_fake, loss_D


def backward_D_basic(netD, real, fake):
    """Calculate GAN loss for the discriminator
    Parameters:
        netD (network)      -- the discriminator D
        real (tensor array) -- real images
        fake (tensor array) -- images generated by a generator
    Return the discriminator loss.
    We also call loss_D.backward() to calculate the gradients.
    """
    # Real
    pred_real = netD(real)
    loss_D_real = GANLoss(pred_real, True)
    # Fake
    pred_fake = netD(fake.detach())
    loss_D_fake =GANLoss(pred_fake, False)
    # Combined loss and calculate gradients
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    loss_D.backward()
    return loss_D_real,loss_D_fake,loss_D
    # return loss_D_real,loss_D_fake,loss_D


def get_target_tensor(prediction, target_is_real):
    """Create label tensors with the same size as the input.
    Parameters:
        prediction (tensor) - - tpyically the prediction from a discriminator
        target_is_real (bool) - - if the ground truth label is for real images or fake images
    Returns:
        A label tensor filled with ground truth label, and with the size of the input
    """
    if target_is_real:
        target_tensor = torch.tensor(1.0)
    else:
        target_tensor = torch.tensor(0.0)
    return target_tensor.expand_as(prediction).to(device)


def GANLoss(prediction, target_is_real):
    target_tensor = get_target_tensor(prediction, target_is_real)
    criterionGAN = nn.BCELoss()
    # criterionGAN = nn.BCEWithLogitsLoss()
    loss = criterionGAN(prediction, target_tensor)
    return loss


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array
    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def print_numpy_to_log(x, f, note):
    x = x.astype(np.float64)
    x = x.flatten()
    print('%s:  mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (note,
                                                                                      np.mean(x), np.min(x), np.max(x),
                                                                                      np.median(x), np.std(x)), file=f)

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        # m.bias.data.fill_(0)


import time


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0  # initial time
    tf = time.time()  # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti  # returns the time difference


TicToc = TicTocGenerator()  # create an instance of the TicTocGen generator


# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % tempTimeInterval)


def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        # m.bias.data.fill_(0)



def FSC_MSE_loss(img1,img2,batch_size):
    batch_size = batch_size
    nx = torch.tensor(img1.shape[-2],device=torch.device("cuda:0")); ny = torch.tensor(img1.shape[-1],device=torch.device("cuda:0")); nmax = torch.max(nx,ny); rnyquist = nmax//2
    x = torch.cat((torch.arange(0,nx/2),torch.arange(-nx/2,0))).to('cuda')
    y =  torch.cat((torch.arange(0,ny/2),torch.arange(-ny/2,0))).to('cuda')
    X,Y = torch.meshgrid(x,y)
    map = X**2 + Y**2
    index = torch.round(torch.sqrt(map.float()))
    r = torch.arange(0,rnyquist+1).to('cuda')
    F1 = torch.rfft(img1,2,onesided=False).permute(1,2,0,3)
    F2 = torch.rfft(img2,2,onesided=False).permute(1,2,0,3)
    C_r= torch.empty(rnyquist+1,img1.shape[0]).to('cuda');
    C1 = torch.empty(rnyquist+1,img1.shape[0]).to('cuda'); 
    C2 = torch.empty(rnyquist+1,img1.shape[0]).to('cuda');
    C_i= torch.empty(rnyquist+1,img1.shape[0]).to('cuda');
    for ii in r:
          auxF1 = F1[torch.where(index == ii)]
          auxF2 = F2[torch.where(index == ii)]
          C_r[ii] = torch.sum(auxF1[:,:,0]*auxF2[:,:,0] + auxF1[:,:,1]*auxF2[:,:,1],axis = 0)
          C_i[ii] = torch.sum(auxF1[:,:,1]*auxF2[:,:,0] - auxF1[:,:,0]*auxF2[:,:,1],axis = 0)
          C1[ii] = torch.sum(auxF1[:,:,0]**2 + auxF1[:,:,1]**2,axis = 0)
          C2[ii] = torch.sum(auxF2[:,:,0]**2 + auxF2[:,:,1]**2,axis = 0)

    FSC = torch.sqrt(C_r ** 2 + C_i ** 2) / torch.sqrt(C1 * C2)
    FSCm = 1 - torch.where(FSC != FSC, torch.tensor(1.0, device=torch.device("cuda:0")), FSC)
    My_FSCloss = torch.mean((FSCm) ** 2)
    # My_MSE = torch.mean((img1 - img2)**2)
    # My_L1 = torch.mean(torch.abs(img1 - img2))
    # loss = My_MSE + alp * My_FSCloss
    loss = My_FSCloss
    return loss

def FSC_loss_complex(img1ph,img1at,img2ph,img2at,batch_size):
    # FSC loss for complex tensors
    img1r = img1at*torch.cos(img1ph).squeeze()
    img1i = img1at*torch.sin(img1ph).squeeze()
    img1 = torch.stack((img1r,img1i),-1)

    img2r = img2at*torch.cos(img2ph).squeeze()
    img2i = img2at*torch.sin(img2ph).squeeze()
    img2 = torch.stack((img2r,img2i),-1)

    batch_size = batch_size
    nx = torch.tensor(img1.shape[-1],device=torch.device("cuda:0")); ny = torch.tensor(img1.shape[-1],device=torch.device("cuda:0")); nmax = torch.tensor(img1.shape[-1],device=torch.device("cuda:0")); rnyquist = nx//2
    x = torch.cat((torch.arange(0,nx/2),torch.arange(-nx/2,0))).to('cuda')
    y = x
    X,Y = torch.meshgrid(x,y)
    map = X**2 + Y**2
    index = torch.round(torch.sqrt(map.float()))
    r = torch.arange(0,rnyquist+1).to('cuda')
    F1 = torch.fft(img1,signal_ndim=2).permute(1,2,0,3)
    F2 = torch.fft(img2,signal_ndim=2).permute(1,2,0,3)
    C_r= torch.empty(rnyquist+1,batch_size).to('cuda');C1 = torch.empty(rnyquist+1,batch_size).to('cuda'); C2 = torch.empty(rnyquist+1,batch_size).to('cuda');C_i= torch.empty(rnyquist+1,batch_size).to('cuda');
    for ii in r:
          auxF1 = F1[torch.where(index == ii)]
          auxF2 = F2[torch.where(index == ii)]
          C_r[ii] = torch.sum(auxF1[:,:,0]*auxF2[:,:,0] + auxF1[:,:,1]*auxF2[:,:,1],axis = 0)
          C_i[ii] = torch.sum(auxF1[:,:,1]*auxF2[:,:,0] - auxF1[:,:,0]*auxF2[:,:,1],axis = 0)
          C1[ii] = torch.sum(auxF1[:,:,0]**2 + auxF1[:,:,1]**2,axis = 0)
          C2[ii] = torch.sum(auxF2[:,:,0]**2 + auxF2[:,:,1]**2,axis = 0)

    FSC = torch.sqrt(C_r ** 2 + C_i ** 2) / torch.sqrt(C1 * C2)
    FSCm = 1 - torch.where(FSC != FSC, torch.tensor(1.0, device=torch.device("cuda:0")), FSC)
    My_FSCloss = torch.mean((FSCm) ** 2)
    # My_MSE = torch.mean((img1 - img2)**2)
    # My_L1 = torch.mean(torch.abs(img1 - img2))
    # loss = My_MSE + alp * My_FSCloss
    loss = My_FSCloss
    return loss

def save_model(name,path,epoch,net,optimizer,loss):
    model_save_name = F'{name}_{epoch}epoch.pt'
    # print(path)
    if not os.path.exists(path):
        os.makedirs(path)
    print('saving trained model {}'.format(model_save_name))
    torch.save({
          'epoch': epoch,
          'model_state_dict': net.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': loss}, path+F'/{model_save_name}')

def write_h5(filename, data):
    with h5py.File(filename,'w') as f:
        gr = f.create_group('data')
        gr["data"] = data

def read_h5(filename):
    with h5py.File(filename,'r') as f:
        data = f['data']['data']
    return data[:]
