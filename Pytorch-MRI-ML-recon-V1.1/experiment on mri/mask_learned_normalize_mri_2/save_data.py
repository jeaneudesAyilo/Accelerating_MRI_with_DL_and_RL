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

def make_kspace_data(directory, seed = 229):
    
    ##load and prepare dataset 

    k_space_data = []


    #directories = glob.glob("/data1/home/jean-eudes.ayilo/Pytorch-MRI-ML-recon-V1.1/data/data_for_mask_learning/output/*")
    directories = glob.glob(directory) #"./data/output/ancien/*"

    for i in range(len(directories)):

        print(f"{i} :", "  ",f"{directories[i]}/imc.nii")

        imc = np.transpose(load_complex_nifty(f"{directories[i]}/imc.nii")[:,:,:,0,:], (2,3,0,1)) # nb slice,ncoil, h,w ; en chargeant avec load_complex_nifty,les données sont : (121, 145, 121, 1, 8); donc on enlève le 1 avec [:,:,:,0,:]

        k_space = normalize(transform_image_to_kspace(imc[50:90], dim=(2,3)))
        #k_space = transform_image_to_kspace(imc[50:89], dim=(2,3)) #tranform image to kspace, normalize, and then take the slice 50 to 90        

        k_space_data.append(k_space)                   


    k_space_data = np.vstack(k_space_data) 
    print("k_space_data : ",k_space_data.shape)       

    index = np.arange(k_space_data.shape[0]) 
    np.random.seed(seed)
    np.random.shuffle(index)
    index = list(index)

    k_space_data_schuffled =  k_space_data[index,...]     
        
    return k_space_data_schuffled


#==========

train_directory = "../../data/data_for_mask_learning/output/train/*"
test_directory = "../../data/data_for_mask_learning/output/test/*"
saving_path = "../../data/data_for_mask_learning/output"


k_space_data_train = make_kspace_data(train_directory, seed = 229)
k_space_data_test = make_kspace_data(test_directory, seed = 229)

print("k_space_data_train.shape : " , k_space_data_train.shape)
print("k_space_data_test.shape : ", k_space_data_test.shape)


np.save(os.path.join(saving_path,"k_space_data_train.npy" ) ,k_space_data_train)
np.save(os.path.join(saving_path,"k_space_data_test.npy" ),k_space_data_test)

saving_path = "../../data/data_for_mask_learning/output"
#saving_path = "/data1/home/jean-eudes.ayilo/Pytorch-MRI-ML-recon-V1.1/data/data_for_mask_learning/output"
x_train = np.load(os.path.join(saving_path,"k_space_data_train.npy" ))
x_test = np.load(os.path.join(saving_path,"k_space_data_test.npy" ))

print("x_train.shape : ", x_train.shape)
print("x_test.shape : ",x_test.shape)

print("####### Fin ######")

