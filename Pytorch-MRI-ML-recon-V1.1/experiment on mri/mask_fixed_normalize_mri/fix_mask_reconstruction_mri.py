
import itertools
import math
import time
import numpy as np
from easydict import EasyDict as edict
import tempfile
import pandas as pd
import glob
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

from torchsummary import summary
from sklearn.preprocessing import MinMaxScaler
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.append('../../')
sys.path.append('../../binary_stochastic_neurons')
sys.path.append('../../utils')

from torch.utils.data import Dataset, DataLoader, TensorDataset 
from utils.load_data import load_complex_nifty
from utils.MRI_util import transform_image_to_kspace, transform_kspace_to_image, transform_kspace_to_image_torch


from binary_stochastic_neurons.utils import Hardsigmoid
from binary_stochastic_neurons.activations import DeterministicBinaryActivation, StochasticBinaryActivation

#==========
use_cuda=True and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
CUDA_LAUNCH_BLOCKING=1

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

#==========

def transform_coil_to_column(kspaces): 
    
    """kspaces : (nslices, ncoils, h, w)
    this function will stack the coil of all slice into columns. eg : it will return a 2D array of shape [n , ncoils], """
    assert kspaces.ndim ==4
    
    b = kspaces.reshape(kspaces.shape[0], kspaces.shape[1], -1)
    
    return np.stack( [np.hstack( [ b[i,j,:] for i in range(b.shape[0]) ] ) for j in range(b.shape[1]) ],axis =1)


def normalize( data, mse_loss_type ='norm', scaling_factor = 1e3):  #from mriGANdata.py : normalize(self, data)    
    '''
        Options: 
            norm => divide by norm of the signal
            raki => multiply by 0.015/max(data)
            weight => unnormalized data
        returns the normalized input
    '''
    if(mse_loss_type == 'norm'):
        normalize = 1/np.max(abs(data[:])) #1/np.linalg.norm((data))
    else:
        normalize = scaling_factor

    norm_data = np.multiply(data,normalize) 
    return norm_data

#==========

class own_minmaxscaler:

    def __init__(self, config, base_scaler=MinMaxScaler()):
        self.base_scaler = base_scaler
        self.config = config


    def fit(self,train_data):
                       
        assert train_data.ndim ==2  and  train_data.shape[1]== self.config.ncoils  and  np.iscomplexobj(train_data)==False ## we want x_train to be of shape [n,ncoils] ==> the complexe [N,ncoils,121,145] should be set to this shape
        self.base_scaler.fit( np.abs(train_data)  ) ## get the min and max by doing : base_scaler.data_max_ ; base_scaler.data_min_

    def transform(self,x):
        assert x.ndim == 4 and x.shape[1] == self.config.ncoils
        
        max_in_coil =(self.base_scaler.data_max_).reshape(1, self.config.ncoils, 1, 1)
        return x/max_in_coil   

    def inverse_transform(self,x):
        
        max_in_coil =(self.base_scaler.data_max_).reshape(1, self.config.ncoils, 1, 1)
        return x* max_in_coil

    
    
def transform_data(x_train, x_test,config, scaler):      
    
    config = edict(config)
    ##x_train, x_test are typically  the raw mnist data divided by 255 : X_train_mnist and X_test_mnist or the kspace data [nslices, ncoils, h,w]
        
    
    train_size = x_train.shape[0]
    test_size= x_test.shape[0]
    

    complex_coil_column = transform_coil_to_column(x_train) ##the column here are complex value

    u_train =  np.vstack( (complex_coil_column.real, complex_coil_column.imag) ) 

    ##normalize kspace
    scaler.fit(u_train); x_train = scaler.transform(x_train) ; x_test= scaler.transform(x_test)

    print("max on train data : ",scaler.base_scaler.data_max_,"\n","min on train data : ", scaler.base_scaler.data_min_ )

    ##convert to torch tensor
    x_train = torch.stack((torch.from_numpy(x_train.real), torch.from_numpy(x_train.imag)),dim=-1).float().to(device) 
    x_test =  torch.stack((torch.from_numpy(x_test.real), torch.from_numpy(x_test.imag)),dim=-1).float().to(device)

    print("x_train shape :", x_train.shape)
    
    dataset = customDataset(x_train, torch.ones((train_size,1,1,1)).float().to(device))
    train_loader = DataLoader(dataset, batch_size=config.batch_size_train, shuffle=True )

    dataset_test = customDataset(x_test , torch.ones((test_size,1,1,1)).float().to(device))
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size_test, shuffle=False )

    return train_loader, test_loader , scaler ##I return the scale , so that in prediction time I could rescale the predicted values    


#=======Data loading

saving_path = "../../data/data_for_mask_learning/output"

k_space_data_train = np.load(os.path.join(saving_path,"k_space_data_train.npy" ))
k_space_data_test = np.load(os.path.join(saving_path,"k_space_data_test.npy" ))

print(k_space_data_train.shape)
print(k_space_data_test.shape)

#==========

class customDataset(Dataset):
    ''' 
        Create a dataloader which takes a tuples of kspaces and one fully sampled image

        Args:
        Kspace: kspace undesampled data
        acs: autocalibration data for training
        im_fs: fully sampled image for adversarial training
        mask: undersampling mask
    '''
    #

    def __init__(self, kspace, input_1):
        self.kspace = kspace
        self.input_1 = input_1
        #self.img_sos = img_sos

        
    def __getitem__(self, index):
        sample = {'kspace': self.kspace[index], 'input_1': self.input_1[index]} #, 'img_sos' : self.img_sos[index]}
        return sample
    
    def __len__(self):
        return len(self.kspace)
    

#==========Defining some layers/functions
# https://github.com/a554b554/kWTA-Activation/blob/master/kWTA/models.py

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
        
        self.k = int((self.slice_dim[0]*self.slice_dim[1])*self.acc)  
        self.k_per_under_mask = math.ceil((self.height_mask)*(self.width_mask) * self.k / (self.slice_dim[0]*self.slice_dim[1]))  
                                              
            
    def forward(self, x):#, k=0):
        #if (k==0):
        #  k = self.k
        #else:
        #  self.k = k
        topval = x.topk(self.k_per_under_mask, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1 ,0)
        comp = ( x>=topval).to(x)
        return comp *x


#==========


from torch.nn import Module, Sequential, ModuleList
from graphs.models.custom_layers.complex_layers import *


#SPIRiT as single conv layer with kernel(centre) = 0
class spirit2D(Module):
    
    def __init__(self, config):
        
        super(spirit2D,self).__init__()
        
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

    
    
class SpiritConvBlock(Module):
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
class stackSpirit(Module):
    def __init__(self, config):
        
        super().__init__() 

        #define local variables
        self.config = config 
        
        #N times spirit is apply
        self.num_stacks = self.config.spirit_block
        
        self.body = ModuleList(
            [SpiritConvBlock(spirit2D(config)) for _ in range(self.num_stacks)]
        )

    def forward(self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,):
        
        kspace_pred = masked_kspace.clone()

        for stack in self.body:
            kspace_pred = stack(kspace_pred, masked_kspace, mask)
        
        return kspace_pred


    
##==========Model definition and training


def test_spirit_2(network,criterion,config, loader, scaler, fix_mask=False):

  network.eval()
  test_loss = 0
  output_array =[]
  input_array = []

  #print("*********** TEST *************")
  with torch.no_grad():
    for batch_idx, sample in enumerate(loader):
      input,  ones_data = sample['kspace'], sample['input_1']
    
      if fix_mask == False :
        output,  learned_mask_adj, learned_mask = network(input,ones_data) ##here the output is image
      else:
        output,  learned_mask_adj, learned_mask = network(input) ##here the output is image
      
      #print(f"test : ,{len(output)}")

      if config.normalized_mse == True:
        test_loss += criterion(output, input)/criterion(input,torch.zeros(input.shape).cpu().to(device)).item() ## I am using this formulation to be sure that I divide the sum of sq by the same denominator            

      else :
        test_loss += criterion(output, input).cpu().item()*input.size(0) #criterion_MSE(output, input).item()*input.size(0)

      output_array.append(output) 
      input_array.append(input)        

  if config.normalized_mse == True:
    test_loss /= len(loader)
  else:
    test_loss /= len(loader.dataset) ##this is the loss on the whole test set

    
  print('\nTest set: Avg. loss: {:.4f}'.format(test_loss))

  input_array = np.array(torch.vstack(input_array).cpu()) 
  output_array = np.array(torch.vstack(output_array).cpu()) 
    

  input_array_complex = input_array[...,0] +1j*input_array[...,1] ; input_array_complex = scaler.inverse_transform(input_array_complex) ##rescaling
  input_array = transform_kspace_to_image(input_array_complex, dim=(2,3))
  input_array = np.sqrt(np.sum(np.square(np.abs(input_array)),axis=(1)))
  
  output_array_complex = output_array[...,0] +1j*output_array[...,1] ; output_array_complex = scaler.inverse_transform(output_array_complex) ##rescaling
  output_array = transform_kspace_to_image(output_array_complex, dim=(2,3))
  output_array = np.sqrt(np.sum(np.square(np.abs(output_array)),axis=(1)))
    
    
  if fix_mask == False :
    fig, ax = plt.subplots(1,3, figsize=(15,6))
    ax[0].imshow(network.my_conv_mask.conv_mask.weight.data.clone().cpu()[0,0,:,:],cmap = "Greys_r")
    #ax[0].axis("off")
        
    #print(learned_mask.clone().cpu().shape)
    ax[1].imshow(learned_mask.clone().cpu()[0,0,:,:],cmap = "Greys_r")
    ax[1].axis("off")
    
    ax[2].imshow(learned_mask_adj.clone().cpu()[0,0,:,:],cmap = "Greys_r")
    ax[2].axis("off")
    plt.show()

  else :
    fig, ax = plt.subplots(1,2, figsize=(12,6))
        
    ax[0].imshow(learned_mask.clone().cpu()[0,0,:,:],cmap = "Greys_r")
    ax[0].axis("off")
    
    ax[1].imshow(learned_mask_adj.clone().cpu()[0,0,:,:],cmap = "Greys_r")
    ax[1].axis("off")
    plt.show()

  return input_array, output_array, test_loss.item(), learned_mask_adj 


##=======

def get_initial_weight_mask(network, data_loader):

  network.eval()

  with torch.no_grad():
    one_batch = next(iter(data_loader))    
    input, ones_data = one_batch['kspace'], one_batch['input_1']
    init_output, init_learned_mask_adj, init_learned_mask = network(input,ones_data)
    print(input.shape)


  return network.my_conv_mask.conv_mask.weight.data.clone().cpu()[0,0,:,:], init_learned_mask_adj.clone().cpu()[0,0,:,:]


def get_std_noise(current_epoch, std_start,std_end, std_decay, std_slope = 1):    
    return std_end + (std_start - std_end) * math.exp(-std_slope * current_epoch / std_decay)


def train_spirit_2(n_epochs, network, optimizer, config,loader, loader_test , scaler,saving_path = None,criterion=nn.MSELoss().to(device),fix_mask = False ):  
          
                
    train_losses = []
    test_losses = []
    learned_mask_adj_list = []
    weight_list = []
  
    if fix_mask == False :
        init_weight, init_learned_mask_adj = get_initial_weight_mask(network, loader) 

        weight_list.append(init_weight); learned_mask_adj_list.append(init_learned_mask_adj)

        plt.imshow(init_learned_mask_adj,cmap = "Greys_r")
        plt.axis("off")
        plt.title("initialisation mask")
        try:
            plt.savefig(os.path.join(saving_path,"initialisation_mask.png")) 
        except:
            pass    

    ##training loop
    for epoch in range(1, n_epochs + 1):
                
        running_loss = 0.0  
        
        if network.config.decrease_std:
            network.std_noise = get_std_noise(epoch, epsilon_start=config.epsilon_start, epsilon_end = config.epsilon_end, eps_decay= config.eps_decay, std_slope = config.std_slope)
            
            print(f"epoch {epoch} : std_noise : {network.std_noise}")
            
        
        network.train()
        for batch_idx, sample in enumerate(loader):                
            input, ones_data = sample['kspace'], sample['input_1']
            optimizer.zero_grad()

            if fix_mask == False :
              output,  learned_mask_adj, learned_mask = network(input,ones_data)
            else :
              #print("input",input.get_device())
              output,  learned_mask_adj, learned_mask = network(input)

            #loss = criterion(output, input)      #(sum of square)/batch_size*C*h*w

            if config.normalized_mse == True:
              loss = criterion(output, input)/criterion(input,torch.zeros(input.shape).to(device)) ## I am using this formulation to be sure that I divide the sum of sq by the same denominator            
            else :
              loss = criterion(output, input)  
            
            loss.backward()
            optimizer.step()

            if fix_mask == False :
              weight_list.append(network.my_conv_mask.conv_mask.weight.data.clone().cpu()[0,0,:,:]) ; learned_mask_adj_list.append(learned_mask_adj.clone().cpu()[0,0,:,:])

            #print("nb of pixels changed wrt previous mask : ", (torch.eq(learned_mask_adj_list[-2], learned_mask_adj_list[-1])==False).sum().item() )                        
            #print("nb of pixels changed wrt the initial mask : ", (torch.eq(learned_mask_adj_list[0], learned_mask_adj_list[-1])==False).sum().item()  )
            

            if config.normalized_mse == True:
              running_loss += loss.cpu().item()
            else:
              running_loss += loss.cpu().item() * input.size(0)  #(sum of square)/C*h*w
            
             

            if batch_idx % 64 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, batch_idx * len(input), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item())) # this is the loss for a single batch : (sum of square)/batch_size*C*h*w
                
                #inp, out, _, _ = test_spirit_2(network,criterion,config, loader = loader_test,scaler = scaler,fix_mask=fix_mask) ; plotting_reconst(inp, out,n=4)


        if config.normalized_mse == True:
          epoch_loss = running_loss/len(loader)
        else:
          epoch_loss = running_loss/len(loader.dataset) #(sum of square)/n_examples*C*h*w; n_examples=batch_size*len(train_loader.dataset)

        train_losses.append(epoch_loss)
        print("test at the end of epoch")
        
        inp_, out_, end_test_loss, epoch_learned_mask_adj = test_spirit_2(network,criterion,config, loader = loader_test,scaler = scaler, fix_mask=fix_mask) ; test_losses.append(end_test_loss) 
        
        plt.imshow(epoch_learned_mask_adj[0,0,:,:].cpu(),cmap = "Greys_r")
        plt.axis("off")
        plt.title(f"mask_at_the_epoch_{epoch}")
        try:
          plt.savefig(os.path.join(saving_path,f"mask_at_the_epoch_{epoch}.png")) 
        except:
          pass

    if fix_mask == False :
      delta_pixels = [(torch.eq(learned_mask_adj_list[i], learned_mask_adj_list[i+1])==False).sum().item() \
                      for i in range(len(learned_mask_adj_list)-1)] #list of nb of pixels changed wrt previous mask 

      delta_weight = [((torch.linalg.vector_norm(torch.flatten(weight_list[i+1]-weight_list[i])))**2).item() \
          for i in range(len(weight_list)-1)] #list of diff in weight wrt previous weight


      delta0_pixels = [(torch.eq(learned_mask_adj_list[0],learned_mask_adj_list[i])==False).sum().item() \
                      for i in range(1,len(learned_mask_adj_list))] #nb of pixels changed wrt the initial mask

      delta0_weight = [((torch.linalg.vector_norm(torch.flatten(weight_list[i]-weight_list[0])))**2).item() \
                for i in range(1,len(weight_list))]  #list of diff in weight wrt initial weight


      fig, ax = plt.subplots(2,2, figsize=(20,10))

      ax[0,0].plot(range(1,len(delta_pixels)+1), delta_pixels)
      ax[0,0].set_title("nb of pixels changed wrt previous mask")
      ax[0,0].set_xlabel("iterations") # batch_size*nb_epochs
      ax[0,0].set_ylabel("nb of pixels")
      
      ax[0,1].plot(range(1,len(delta_weight)+1),delta_weight)
      ax[0,1].set_title("square diff in weight wrt previous weight")
      ax[0,1].set_xlabel("iterations") 
      ax[0,1].set_ylabel("weight square diff")

      ax[1,0].plot(range(1,len(delta0_pixels)+1),delta0_pixels)
      ax[1,0].set_title("nb of pixels pixels changed wrt the initial mask")
      ax[1,0].set_xlabel("iterations") 
      ax[1,0].set_ylabel("nb of pixels")

      ax[1,1].plot(range(1,len(delta0_weight)+1), delta0_weight)
      ax[1,1].set_title("diff in weight wrt initial weight")
      ax[1,1].set_xlabel("iterations") 
      ax[1,1].set_ylabel("weight square diff")

      try:
        plt.savefig(os.path.join(saving_path,"variations.png")) 
      except:
        pass
      plt.show() 

    return {"train_loss":train_losses, "test_loss":test_losses}  



def plotting_reconst(groundtruth, reconstruction,mask, loader,n=3, seed= None, saving_path =None):
  ##choose n between 3 and 6
  if seed != None:
    np.random.seed(seed)     
  random_index = np.random.choice(len(loader.dataset), n)

  fig, ax = plt.subplots(2,n, figsize=(15,6))
    #plot some examples of reconstruction and their groundtruth
  for t in range(n):

        ax[0,t].imshow(groundtruth[ random_index[t],:,:],cmap = "Greys_r")  
        ax[0,t].set_title("groundtruth")
        ax[0,t].axis("off")    

        ax[1,t].imshow(reconstruction[ random_index[t],:,:],cmap = "Greys_r")
        ax[1,t].set_title("reconstruction")
        ax[1,t].axis("off")
  try:
    plt.savefig(saving_path)
  except:
    pass
  plt.show()
    

    
def compute_ssim(groundtruth, reconstruction):    

    a = np.moveaxis(groundtruth, 1, -1) ## groundtruth dimension is (N,1,img_h,img_w); turn it to (N,img_h,img_w,1)
    b =  np.moveaxis(reconstruction, 1, -1)
    ssim_list = []
    psnr_list = []
    for i in range(a.shape[0]):
         ssim_i = ssim(a[i], b[i], data_range=abs(a[i].max() - b[i].min()), multichannel=True )
         psnr_i = psnr(a[i], b[i], data_range=abs(a[i].max() - b[i].min()))
    ssim_list.append(ssim_i)
    psnr_list.append(psnr_i)
    
    ##I double compute, just for comparison
    #ssim_2 = torch_ssim(torch.from_numpy(groundtruth),torch.from_numpy(reconstruction)); psnr_2 = torch_psnr(torch.from_numpy(groundtruth),torch.from_numpy(reconstruction))
    return {"test_ssim": np.mean(ssim_list), "test_psnr": np.mean(psnr_list)}#,"test_ssim2": ssim_2, "test_psnr2":psnr_2 }
    


def plot_loss(dictionay,title="reconstruction loss", test=False, saving_path = None):
  print("dictionay['test_loss']", dictionay["test_loss"])
  plt.figure()
  plt.plot(np.arange(1, len(dictionay["train_loss"])+1), dictionay["train_loss"], label = "train loss") 
  if test==True:
    plt.plot(np.arange(1, len(dictionay["test_loss"])+1), dictionay["test_loss"], label = "test loss")
  plt.title(f"{title}")
  plt.ylabel("loss")
  plt.xlabel("epoch") 
  plt.legend()
  try:
    plt.savefig(saving_path) 
  except:
    pass

  plt.show()
                
        
#===========
def create_mask(size, acceleration, typ, submask_size = (4,4),seed = None):
    """acceleration ici correspond au nombre de points  """
    num_cols = size[-1] 
    num_rows = size[-2]
    
        #acs_start = (num_cols - acs_lines + 1) // 2

    if typ=='uniform':

        mask_col = np.zeros(num_cols, dtype=bool)
        mask_col[::acceleration*2] = True
        ones = np.ones((num_rows,num_cols))
        mask = np.multiply(ones, mask_col)

    if typ == "2D_uniform":

        mask_row = np.zeros(num_rows, dtype=bool)
        mask_row[::acceleration] = True 
        mask_row ## indicates which lines in the first column are set to true or false

        mask_row = mask_row[:,None]
        mask_row

        mask_col = np.zeros(num_cols ,dtype=bool) ##indicate if each column has some acquired points or not at all
        mask_col[::acceleration] = True
        mask_col = np.tile(mask_col,(num_rows,1))
        mask_col

        mask = np.multiply(mask_row, mask_col)
        
    
    if typ == "caipiranha":

        ## caipiranha sampling mask , see https://mriquestions.com/caipirinha.html

        ##the goal is to build a small matrix (shape :(num_rows,accelaration)) which has a caipiranha pattern and then apply it to shape of the data

        mask_row = np.zeros(num_rows ,dtype=bool)
        mask_row[::acceleration] = True 
        mask_row = mask_row[:,None] ## indicates which lines in the first column are set to true or false

        mask_col = np.zeros(acceleration*2 ,dtype=bool)
        mask_col[::acceleration] = True
        mask_col = np.tile(mask_col,(num_rows,1))
        mask_col  ##assign true or false to each column

        dd = np.multiply(mask_row, mask_col) ##replecate the first column at columns which are set to true. dd is our small matrix that has a caipiranha pattern

        replace=(dd == False)[:,acceleration]  ## make a sampling shift : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3865024/ paragraph CAIPIRINHA Theory

        dd[:,acceleration] = replace

        mask = np.tile(dd,int(np.ceil(num_cols/dd.shape[1]))) ##replicate this matrix 

        mask = mask[:,:num_cols] #apply it to shape of the data
        #mask

    if typ == "random":

      k_per_under_mask = (submask_size[0]*submask_size[1])//4

      np.random.seed(seed)

      ##for the proportion we need to find the proportion of 1 we will have in a reference submask
      #https://stackoverflow.com/questions/19597473/binary-random-array-with-a-specific-proportion-of-ones

      #mask = np.random.choice([1, 0], size= size, p=[(acceleration*acceleration)/(submask_size[0]*submask_size[1]) , 1 - (acceleration*acceleration)/(submask_size[0]*submask_size[1])] )

      mask = np.array( [0] * (submask_size[0]*submask_size[1] - k_per_under_mask) + [1] * (k_per_under_mask) )
      
      np.random.shuffle(mask); mask = mask.reshape((submask_size[0],submask_size[1])); 

      mask = np.tile(mask, (math.ceil(num_rows/submask_size[0]),math.ceil(num_cols/submask_size[1])))[:num_rows,:num_cols] 

    return mask



#==========Model definition and training       
class FixmaskNet(nn.Module):

    def __init__(self, masque, config=None):
        super(FixmaskNet, self).__init__()

        self.img_h = config.slice_dim[0]
        self.img_w = config.slice_dim[1]

        self.height_mask = masque.shape[0]
        self.width_mask = masque.shape[1]
        
        self.acs = int(config.acs_full * 1/(config.acceleration*2)) ##here in fixed pattern we consider just an acceleration of 4 ie 2 on column and 2 on axis ==> 1/(config.acceleration*2) means a proportion of 25%
        
        self.config = config

        self.spirit_nn = stackSpirit(self.config)
        
        self.masque = torch.reshape(torch.from_numpy(masque), (1, 1, self.height_mask, self.width_mask))                       
        
        self.masque_adj = torch.tile(self.masque[:,:,], (math.ceil(self.img_h/self.height_mask),math.ceil(self.img_w/self.width_mask)))[:,:,:self.img_h,:self.img_w] # x shape :[batch_size,n_channel,h,w] ; x.shape[-2] =x.shape[-1] = 28

        
        if self.config.acs_type == "column":
          
            self.masque_adj[:, :, :, self.img_w//2 - self.acs//2 : self.img_w//2 + self.acs//2 +1] = 1 ##acs columns; with this large number the sigmoide should be close to 1
                #print("acs nb col :", (self.w//2 + self.acs//2 +1) - (self.w//2 - self.acs//2))
                
        elif self.config.acs_type == "square":
      
            self.masque_adj[:, :, self.img_h//2 - self.acs//2 : self.img_h//2 + self.acs//2 +1, self.img_w//2 - self.acs//2 : self.img_w//2 + self.acs//2 +1] = 1                                        
                #print("acs shape :", (self.h//2 + self.acs//2 +1) - (self.h//2 - self.acs//2) , " ", (self.w//2 + self.acs//2 +1)- (self.w//2 - self.acs//2))
                
        elif self.config.acs_type == "no": 
            pass

        self.stack_binary_mask_adj = torch.stack((self.masque_adj, self.masque_adj), dim=-1,).float().to(device)

        
    def forward(self, x):

        x  = torch.mul(self.stack_binary_mask_adj, x)

        reconstructed_img = self.spirit_nn(x, self.stack_binary_mask_adj) ##more precisely it is the reconstructed kspace  
          
        return reconstructed_img , self.masque_adj, self.masque  

    
    
##=============
    
def run_fix_mask_model_2(config, trainset= k_space_data_train, testset = k_space_data_test, critere = nn.MSELoss().to(device)):
    
    config = edict(config.copy()) 
    
    dir_name = f"acs_type_{config.acs_type}_mask_typ_{config.mask_typ}_mask_seed_{config.mask_seed}_mask_dim_{config.mask_dim[0]}_{config.mask_dim[1]}_spirit_bloc_{config.spirit_block}_kernel_{config.kernel1[0]}_{config.kernel1[1]}_lr_other_{config.lr_other}"
    
    
    try:
        os.mkdir(config.main_directory) ##this will be created once, the others times an error will occur due to the multiplication of config that will need to create it again
    except:
        pass
        

    try:
        save_path = os.path.join(config.main_directory, dir_name)
        os.mkdir(save_path)  
    except:
        pass


    mask = create_mask(config.mask_dim, acceleration=2, typ=config.mask_typ, submask_size = (4,4),seed = config.mask_seed)
    
    print("#################################################################")                          

    print(f"acs_type_{config.acs_type}_mask_typ_{config.mask_typ}_mask_seed_{config.mask_seed}_mask_dim_{config.mask_dim}_spirit_bloc_{config.spirit_block}_lr_other_{config.lr_other}_kernel_{config.kernel1}")
    
    
    df = pd.DataFrame(columns=['acs_type','mask_typ','mask_seed','mask_dim','bloc', 'kernel','lr_other','test_ssim','test_psnr','test_loss','test_loss_denorm'])
    
        
    ##initialize scaler
    init_scaler = own_minmaxscaler(config)
    
    train_loader, test_loader, input_scaler = transform_data(trainset, testset,config, init_scaler)                          
        
    reconst_network = FixmaskNet(mask, config = config).to(device)    

        ##Initialize spirit weights
    for m in reconst_network.modules():  ## from self.netS.apply(weights_init) mriPatchGanSpirit.py ; Pytorch-MRI-ML-recon-V1.1/graphs/weights_initializer.py Swetali codes
    
        if (type(m) in [nn.Conv2d]): 
            m.weight.data.normal_(0.0, 0.02)
    
    
    optimizer = optim.Adam(reconst_network.parameters(), lr= config.lr_other)
    
    print("start running")

    history = train_spirit_2(config.n_epochs, network = reconst_network,optimizer = optimizer,config = config, loader =train_loader, loader_test= test_loader, scaler = input_scaler, criterion=critere, saving_path = save_path ,fix_mask = True)
    
    plot_loss(history, title=f"reconstruction loss_spirit_bloc_{config.spirit_block}_lr_other_{config.lr_other}_kernel_{config.kernel1[0]}_{config.kernel1[1]}", test=True, saving_path = os.path.join(save_path,"loss_curve.png") )

    test_input, test_output,test_loss, finalmask = test_spirit_2(reconst_network,criterion=critere, config =config, loader = test_loader,scaler = input_scaler, fix_mask = True)    


    plotting_reconst(test_input , test_output,mask =finalmask,loader=test_loader, n=4, seed= 815, saving_path =os.path.join(save_path,"reconst_examples1.png"))
    
    plotting_reconst(test_input , test_output,mask =finalmask,loader=test_loader, n=4, seed= 23, saving_path =os.path.join(save_path,"reconst_examples2.png"))

    plotting_reconst(test_input , test_output,mask =finalmask,loader=test_loader, n=4, seed= 100, saving_path =os.path.join(save_path,"reconst_examples3.png"))

    metrics = compute_ssim(test_input ,test_output)
    
    test_loss_denorm = critere(torch.from_numpy(test_input), torch.from_numpy(test_output)).item()

    print("average ssim and psnr on test set", metrics)    
    
    line ={"acs_type":config.acs_type,"mask_typ":config.mask_typ,"mask_seed":config.mask_seed ,"mask_dim":config.mask_dim ,"bloc" : config.spirit_block, "kernel" :config.kernel1,"lr_other": config.lr_other}

    line.update(metrics); line["test_loss"] = test_loss ; line["test_loss_denorm"] = test_loss_denorm

    df = df.append(line, ignore_index=True) 

    df.to_csv(os.path.join(config.main_directory, dir_name ,"result_grid_search.csv"), sep='\t',index=True)           
    

    print("##########   FIN   ##########")
        
         
            
##==============================================================================

import json 

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = edict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)

            
import argparse
# parse the path of the json config file
arg_parser = argparse.ArgumentParser(description="")
arg_parser.add_argument(
    'config',
    metavar='config_json_file',
    default='None',
    help='The Configuration file in json format')
args = arg_parser.parse_args()

# parse the config json file
config,_ = get_config_from_json(args.config)

if __name__ == '__main__':
    run_fix_mask_model_2(config)