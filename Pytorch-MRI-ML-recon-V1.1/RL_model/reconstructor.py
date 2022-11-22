import itertools
import math
import time
import numpy as np
from easydict import EasyDict as edict
import tempfile
import pandas as pd
import os
import sys

import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset 
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

#for colab
#os.chdir("/content/gdrive/MyDrive/Stage_M2_Jean-Eudes/")
#sys.path.append('/content/gdrive/MyDrive/Stage_M2_Jean-Eudes/binary_stochastic_neurons/')
#sys.path.append('/content/gdrive/MyDrive/Stage_M2_Jean-Eudes/binary_stochastic_neurons/distributions/')

##on local
#os.chdir("/content/gdrive/MyDrive/Stage_M2_Jean-Eudes/")
sys.path.append('../')
sys.path.append('../binary_stochastic_neurons/')
sys.path.append('../binary_stochastic_neurons/distributions/')


### Import packages from https://github.com/Wizaron/binary-stochastic-neurons
from activations import DeterministicBinaryActivation, StochasticBinaryActivation
from utils import Hardsigmoid


class SparsifyBase(nn.Module):
    def __init__(self, sparse_ratio=0.5):
        super(SparsifyBase, self).__init__()
        self.sr = sparse_ratio
        self.preact = None
        self.act = None

    def get_activation(self):
        def hook(model, input, output):
            self.preact = input[0].cpu().detach().clone()
            self.act = output.cpu().detach().clone()
        return hook

    def record_activation(self):
        self.register_forward_hook(self.get_activation())      
    

class Sparsify1D_kactiveIOnline(SparsifyBase):
    def __init__(self, config):                                          
        super(Sparsify1D_kactiveIOnline, self).__init__()
        
        self.height_mask = config.mask_dim[0]
        self.width_mask = config.mask_dim[1]
        self.acc = config.acc ## acceleration en proportion
        self.slice_dim = config.slice_dim #tuple or list of the height and width of k-space
        self.D2sampler = config.D2sampler
        
        if self.D2sampler:

          self.k = int((self.slice_dim[0]*self.slice_dim[1])*self.acc)  
          self.k_per_under_mask = math.ceil((self.height_mask)*(self.width_mask) * self.k / (self.slice_dim[0]*self.slice_dim[1]))  
        
        else :        
          self.k = int((self.slice_dim[1])*self.acc)
          self.k_per_under_mask = self.k
            
    def forward(self, x):#, k=0):
        #if (k==0):
        #  k = self.k
        #else:
        #  self.k = k
        topval = x.topk(self.k_per_under_mask, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1 ,0)
        comp = ( x>=topval).to(x)
        return comp *x



class complexConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,dilation=1, groups=1, bias=False):
        
        super(complexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self,input_r, input_i):
        return self.conv_r(input_r)-self.conv_i(input_i), \
               self.conv_r(input_i)+self.conv_i(input_r)


#SPIRiT as single conv layer with kernel(centre) = 0
class Complexspirit2D(nn.Module):
    
    def __init__(self, config):
        
        super(Complexspirit2D,self).__init__()
        
        #define local variables
        self.config = config
        self.kernel_size = self.config.kernel1 
        self.ncoils = self.config.ncoils
        
        #self.nslices = config.batch_size #choosen_batch_size #self.config.nslices # REMPLACER PAR choosen_batch_size  divisible par le nombre d'exemple d'appprentissage total; le batch_size doit être = nb slice à processer
        self.conv1 = complexConv2d(in_channels= self.ncoils , out_channels=self.ncoils, kernel_size=self.kernel_size, bias=False, padding=(self.kernel_size[0]//2, self.kernel_size[1]//2))
    
    def forward(self, x):
        
        (x_real,x_img) = x[...,0],x[...,1]  
        
        with torch.no_grad():
            self.conv1.conv_r.weight[:, :, self.kernel_size[0]//2, self.kernel_size[1]//2] = 0
            self.conv1.conv_i.weight[:, :, self.kernel_size[0]//2, self.kernel_size[1]//2] = 0
        
        (x_real,x_img) = self.conv1(x_real,x_img)
        (x_real,x_img) =(torch.unsqueeze(x_real, 4), torch.unsqueeze(x_img, 4))
        
        return torch.cat((x_real,x_img),-1)

    
    
class ComplexSpiritConvBlock(nn.Module):
    """
    Model block for spirit network.
    This model applied spirit to undersampled data. A series of these blocks can be stacked to form
    the full network.
    """

    def __init__(self, model):
        """
        Args:
            model: Spirit model.
        """
        super().__init__()

        self.model = model

    def forward(self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor):
        
        x = self.model(current_kspace)
        #data consistency
        out = torch.multiply(x,1-mask) + torch.mul(ref_kspace, mask) #torch.multiply(x,~mask) + ref_kspace
        
        return out
    
    
#Stack SpiritConv N times
class ComplexstackSpirit(nn.Module):
    def __init__(self, config):
        
        super().__init__() 

        #define local variables
        self.config = config 
        
        #N times spirit is apply
        self.num_stacks = self.config.spirit_block
        
        self.body = nn.ModuleList(
            [ComplexSpiritConvBlock(Complexspirit2D(config)) for _ in range(self.num_stacks)]
        )

    def forward(self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,):
        
        kspace_pred = masked_kspace.clone()

        for stack in self.body:
            kspace_pred = stack(kspace_pred, masked_kspace, mask)
        
        return kspace_pred

    
class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            #return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
            return din + torch.autograd.Variable(torch.randn(din.size()).to(device) * self.stddev)
        return din    
    

class Net(nn.Module):

    def __init__(self, bin_act_type = "determinist", k_per_under_mask = 2,config=None):
        super(Net, self).__init__()

        self.img_h = config.slice_dim[0] ##nb rows for the inmput image
        self.img_w = config.slice_dim[1] ##nb col for the inmput image
        self.height_mask = config.mask_dim[0]
        self.width_mask = config.mask_dim[1]
        self.k_per_under_mask = k_per_under_mask
        self.config = config
        self.std_noise = config.std_noise
        self.dropout_proba = config.dropout_proba                

        self.bin_act_type = bin_act_type
        
        ##only implemented for full mask
        if self.config.D2sampler == True: 
          self.conv_mask = nn.ConvTranspose2d(in_channels = 1, out_channels =1, kernel_size = (self.height_mask, self.width_mask), groups=1, bias=False)      
        else :##attention , non implementer pour les petits masques
          self.conv_mask = nn.ConvTranspose2d(in_channels = 1, out_channels =1, kernel_size = (1, self.width_mask), groups=1, bias=False) 
        
        
        self.DO1 = nn.Dropout(p = self.dropout_proba, inplace=False)
        self.noise = GaussianNoise(self.std_noise)

        self.binary_act_d = DeterministicBinaryActivation(estimator='ST')
        self.binary_act_s = StochasticBinaryActivation(estimator='ST')
        
        self.slope = config.slope #1.0
        #self.linear_sp = Sparsify1D_kactiveIOnline(self.height_mask, self.width_mask, k_per_under_mask = self.k_per_under_mask)
        self.linear_sp = Sparsify1D_kactiveIOnline(config) 
        print("Number of acquired points in the whole mask :",self.linear_sp.k ); print(f"Number of acquired points in the local mask of size {config.mask_dim} :",self.linear_sp.k_per_under_mask )
        print(f"acceleration : {self.linear_sp.acc}")
        self.spirit_nn = ComplexstackSpirit(self.config)



    def forward(self, x, my_input_1):


          sig_output = F.sigmoid((self.conv_mask(my_input_1)))       
          
          #sig_output= self.DO1(sig_output) # commente dans celui qui fonctionne

          if self.bin_act_type == "determinist": # add noise and use deterministic binary activation with

            sig_output = self.noise(sig_output) # variance plus forte ici
            x2 = sig_output.view(sig_output.size()[0],-1)        
            x2 = self.linear_sp(x2)        
            wta_output = x2.view_as(sig_output) 
                  
            #wta_output = self.linear_sp(sig_output)        
              
            binary_mask = self.binary_act_d([wta_output, self.slope]) # [batch_size,1,self.height_mask, self.width_mask]
          
          elif self.bin_act_type == "stochastic": # add noise and use deterministic binary activation with
            x2 = sig_output.view(sig_output.size()[0],-1)        
            x2 = self.linear_sp(x2)        
            wta_output = x2.view_as(sig_output)    
            binary_mask = self.binary_act_s([wta_output, self.slope]) # [batch_size,1,self.height_mask, self.width_mask], the mask is specific to each input ?


          if not self.config.D2sampler:
            binary_mask = torch.tile(binary_mask[:,:,], (self.img_h ,1)) 


          ##ajuster la taille du masque à celle de l'image                                                                      
          binary_mask_adj = torch.tile(binary_mask[:,:,], (math.ceil(self.img_h/self.height_mask),math.ceil(self.img_w/self.width_mask)))[:,:,:self.img_h,:self.img_w] # x shape :[batch_size,n_channel,h,w] ; x.shape[-2] =x.shape[-1] = 28
          ##or change the "one" input and rewrite the conv mask with convtranspose  ,with stride = dim of mask...
          
          #print("binary_mask_adj shape: ", binary_mask_adj.shape)
          #print("x avant mul shape: ", x.shape)

        
          stack_binary_mask_adj = torch.stack((binary_mask_adj, binary_mask_adj), dim=-1,)
          #print("stack_binary_mask_adj ", stack_binary_mask_adj.shape) ; print("x.shape :", x.shape)
          x  = torch.mul(stack_binary_mask_adj,x)
          reconstructed_img = self.spirit_nn(x, stack_binary_mask_adj) ##more precisely it is the reconstructed kspace
          print()
                    
          return reconstructed_img # binary_mask_adj, binary_mask  
    

reconstructor_config = {'D2sampler': False,
 'acc': 0.25,
 'auxiliary': False,
 'batch_norm': False,
 'dropout_proba': 0.0,
 'kernel1': [5, 5],
 'lr_mask': 0.001,
 'lr_other': 0.01,
 'main_directory': './result',
 'mask_dim': [28, 28],
 'n_epochs': 5,
 'ncoils': 1,
 'normalized_mse': True,
 'shift_type': 'tr_img_ksp',
 'slice_dim': [28, 28],
 'slope': 1,
 'spirit_activation': 'none',
 'spirit_block': 5,
 'std_noise': 0.05}    



supervised_model = Net(bin_act_type = "determinist", config =edict(reconstructor_config))
#supervised_model.load_state_dict(torch.load("/content/single_line_mask_and_spirit.pt"))  
#spirit_reconstructor = supervised_model.spirit_nn.to(self.device)    
