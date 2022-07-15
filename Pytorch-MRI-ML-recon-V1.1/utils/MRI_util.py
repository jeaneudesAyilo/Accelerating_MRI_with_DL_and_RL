from typing import ChainMap
import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn
import numpy as np
import nibabel as ni
import ismrmrd
import ismrmrd.xsd
import os
from utils.load_data import load_complex_nifty
from scipy.optimize import curve_fit
import torch
import torch.nn as nn
import torch.nn.functional as F

#calculate average around the neighbour pixel


def getMeanData(x,type='complex'):
    """
    Calculating the mean of each 3x3 neighborhood.
    input:
      - x: input tensor of dimensions batch-channel-height-width
    output:
      - y: each element in y is the average of the 9 corresponding elements in x
    """
    x_real = torch.unsqueeze(x.real.float(),0)
    x_imag = torch.unsqueeze(x.imag.float(),0)

    # define the filter
    if(len(x.shape)==3):
        weights = torch.ones((3, 3), requires_grad=False)  
        weights = weights / weights.sum()
        weights = weights[None, None, ...].repeat(x.shape[0],1, 1, 1)
        # use grouped convolution - so each channel is averaged separately.  
        y_real = F.conv2d(x_real, weights, padding=1, groups=x.shape[0])
        y_imag = F.conv2d(x_imag, weights, padding=1, groups=x.shape[0]) 
    
    else:
        weights = torch.ones((3, 3, 3), requires_grad=False)  
        weights = weights / weights.sum()
        weights = weights[None, None, ...].repeat(x.shape[0],1, 1, 1, 1)
        # use grouped convolution - so each channel is averaged separately.  
        y_real = F.conv3d(x_real, weights, padding=1, groups=x.shape[0])[0]
        y_imag = F.conv3d(x_imag, weights, padding=1, groups=x.shape[0])[0]  
    
    if(type=='complex'):
        y = torch.stack((y_real,y_imag), -1)
    elif(type=='real'):
       y = torch.cat((y_real,y_imag), 1) 
    else:
        y = y_real + 1j*y_imag

    return y 

#from ismrmrd tool box

def transform_kspace_to_image(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions
    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = range(k.ndim)

    img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    #img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img


def transform_image_to_kspace(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions
    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)

    k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    #k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k

#Special layer for GRAPPA GANS
#to change to SOS imagesc
class kspaceToImageToSOS(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        """ Computes the Fourier transform from k-space to image space
        along a given or all dimensions
        :param k: k-space data
        :param dim: vector of dimensions to transform
        :param img_shape: desired shape of output image
        :returns: data in image space (along transformed dimensions)
        """
        if not self.dim:
            self.dim = range(x.ndim)
        img = torch.fft.ifftshift(x, dim=self.dim)
        img = torch.fft.ifftn(img, dim=self.dim)
        img = torch.fft.fftshift(img,dim=self.dim)
        img = torch.sqrt(torch.sum(torch.square(torch.abs(img)),axis=(1)))
        img = torch.unsqueeze(img,1) #add coil dimesnsion 
        return img

class toReorderGrappa(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, factor, ncoils):
        out = y.clone()
        for i in range(factor-1):     
            out[:,:,:,i+1::factor] = x[:,i*ncoils:(i+1)*ncoils,...].clone()
        
        return out

def transform_kspace_to_image_torch(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions
    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = range(k.ndim)

    img = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(k, dim=dim), s=img_shape, dim=dim), dim=dim)
    #img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img

def transform_image_to_kspace_torch(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions
    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)

    k = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(img, dim=dim), dim=dim), dim=dim)
    #k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k

def ifft(im,axes=(2,3,4)):
    im = np.fft.ifftn(np.fft.ifftshift(im,axes=axes),axes=axes)
    return im

def fft(im,axes=(2,3,4)):
    im = np.fft.fftshift(np.fft.fftn(im,axes=axes),axes=axes)
    return im

#Normalization methods in numpy and torch
def complex_from_two_reals(real,imag):
    result = 1j*imag
    result += real
    return result

def normalize_numpy(datas_numpy):
    datas_numpy_real = datas_numpy.real - datas_numpy.real.mean()
    datas_numpy_imag = datas_numpy.imag - datas_numpy.imag.mean()
    datas_numpy_real = datas_numpy.real / datas_numpy.real.std()
    datas_numpy_imag = datas_numpy.imag / datas_numpy.imag.std()
    return complex_from_two_reals(datas_numpy_real,datas_numpy_imag)

def normalize(datas):
    datas[...,0] -= datas[...,0].mean()
    datas[...,1] -= datas[...,1].mean()
    datas[...,0] /= datas[...,0].std()
    datas[...,1] /= datas[...,1].std()
    return datas

#from ismrmrdtools import show, transform
# The lines are labeled with flags as follows:
# - Noise or Imaging using ACQ_IS_NOISE_MEASUREMENT
# - Parallel calibration using ACQ_IS_PARALLEL_CALIBRATION
# - Forward or Reverse using the ACQ_IS_REVERSE flag
# - EPI navigator using ACQ_IS_PHASECORR_DATA
# - First or last in a slice using ACQ_FIRST_IN_SLICE and ACQ_LAST_IN_SLICE
#refrence script: https://github.com/ismrmrd/ismrmrd-paper/blob/master/code/pybits.py


#function to get acs data
#functions from recon ismrmrd 
def get_acs_data(filename, datasetname='dataset', noise=None, acs_size=40):
    
    # Handle the imaging data
    dset = ismrmrd.Dataset(filename, datasetname, create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc = header.encoding[0]

    # Matrix size
    eNx = enc.encodedSpace.matrixSize.x
    eNy = enc.encodedSpace.matrixSize.y
    eNz = enc.encodedSpace.matrixSize.z
    rNx = enc.reconSpace.matrixSize.x 
    
    # Number of Slices
    if enc.encodingLimits.slice != None:
        nslices = enc.encodingLimits.slice.maximum + 1
    else:
        nslices = 1

    # Loop through the acquisitions ignoring the noise scans
    firstscan = 0
    while True:
        acq = dset.read_acquisition(firstscan)
        if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            firstscan += 1
        else:
            break

    acq = dset.read_acquisition(firstscan)
    ncoils = acq.active_channels
    
    # The calibration data may be have fewer points than the full k
    refNx = acq.number_of_samples
    x0 = int((refNx - refNx/2) / 2)
    x1 = int(refNx - x0)
    
    # Reconsparallel imaging calibration scans
    # Initialiaze a storage array for the reference data
    acs_data = np.zeros((nslices, ncoils, eNx, eNy, eNz), dtype=np.complex64)
    # Loop             
    scan = firstscan
    while True:
        acq = dset.read_acquisition(scan)
        if acq.isFlagSet(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
           
            #remove oversampling 
            xline = transform_kspace_to_image(acq.data, [1])
            xline = xline[:,x0:x1]
            acq.resize(int(refNx/2),acq.active_channels,acq.trajectory_dimensions)
            acq.center_sample = int(refNx/4)
            # need to use the [:] notation here to fill the data
            acq.data[:] = transform_image_to_kspace(xline, [1])
            
            #arranging data
            slice = acq.idx.slice
            y = acq.idx.kspace_encode_step_1
            z = acq.idx.kspace_encode_step_2
            acs_data[slice, :, x0:x1, y, z] = acq.data
            scan += 1
        else:
            break
    dset.close()

    if eNz > 1:
        #3D
        acs_data = acs_data[:,:,x0:x1,0:acs_size,0:acs_size]
    else:
        #2D
        acs_data = acs_data[:,:,x0:x1,0:acs_size,0]
    
    return np.squeeze(acs_data)

#get unsampled data
def get_data(filename, datasetname='dataset', noise=None):
    
    # Handle the imaging data
    dset = ismrmrd.Dataset(filename, datasetname, create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc = header.encoding[0]

    # Matrix size
    eNx = enc.encodedSpace.matrixSize.x
    eNy = enc.encodedSpace.matrixSize.y
    eNz = enc.encodedSpace.matrixSize.z
    rNx = enc.reconSpace.matrixSize.x
    rNy = enc.reconSpace.matrixSize.y
    rNz = enc.reconSpace.matrixSize.z
    
    # Field of View
    eFOVx = enc.encodedSpace.fieldOfView_mm.x
    eFOVy = enc.encodedSpace.fieldOfView_mm.y
    eFOVz = enc.encodedSpace.fieldOfView_mm.z
    rFOVx = enc.reconSpace.fieldOfView_mm.x
    rFOVy = enc.reconSpace.fieldOfView_mm.y
    rFOVz = enc.reconSpace.fieldOfView_mm.z
    
    # Number of Slices, Reps, Contrasts, etc.
    ncoils = header.acquisitionSystemInformation.receiverChannels
    if enc.encodingLimits.slice != None:
        nslices = enc.encodingLimits.slice.maximum + 1
    else:
        nslices = 1
    
    if enc.encodingLimits.repetition != None:
        nreps = enc.encodingLimits.repetition.maximum + 1
    else:
        nreps = 1
    
    if enc.encodingLimits.contrast != None:
        ncontrasts = enc.encodingLimits.contrast.maximum + 1
    else:
        ncontrasts = 1
    
    # Initialiaze a storage array
    all_data = np.zeros((nreps, ncontrasts, nslices, ncoils, int(eNx/2), eNy, eNz), dtype=np.complex64)
    #all_acs = 
    # TODO loop through the acquisitions looking for noise scans
    firstacq=0
    for acqnum in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)
        # TODO: Currently ignoring noise scans
        if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            #print("Found noise scan at acq ", acqnum)
            continue
        elif acq.isFlagSet(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
            #print("Found acs can at acq ", acqnum)
            continue 
        else:
            firstacq = acqnum
            print("Imaging acquisition starts acq ", acqnum)
            break

    x0 = int((eNx - eNx/2) / 2)
    x1 = int(eNx - x0)

    # Loop through the rest of the acquisitions and stuff 
    for acqnum in range(firstacq,dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)

        #remove oversampling 
        xline = transform_kspace_to_image(acq.data, [1])
        xline = xline[:,x0:x1]
        acq.resize(int(eNx/2),acq.active_channels,acq.trajectory_dimensions)
        acq.center_sample = int(eNx/4)
        # need to use the [:] notation here to fill the data
        acq.data[:] = transform_image_to_kspace(xline, [1])
        
        # Stuff into the buffer
        rep = acq.idx.repetition
        contrast = acq.idx.contrast
        slice = acq.idx.slice
        y = acq.idx.kspace_encode_step_1
        z = acq.idx.kspace_encode_step_2
        #print(f"rep: {rep} contrast: {contrast} slice: {slice} y: {y} z: {z}")
        all_data[rep, contrast, slice, :, :, y, z] = acq.data
    print(all_data.shape)

    dset.close()
    if eNz > 1:
        return all_data
    else:
        return all_data[...,:,:,0]

#Energy normalization 
def funcEnergy(x, a, b, c):
    return a * np.exp(-b*np.power(x,2)) + c

def normalizeByEnergy(filename,IM_3D_unseq,acq='2D'):
    extension = os.path.splitext(filename)[1]

    if(extension == '.nii'):
        
        img = ni.load(filename)
        hdr = img.header
        
        if(acq == '3D'):
            # matrix size
            volume_size_1 = hdr['dim'][1:4]

            # voxel dimension
            voxel_size = hdr['pixdim'][1:4]
        else:
            # matrix size
            volume_size_1 = hdr['dim'][1:3]

            # voxel dimension
            voxel_size = hdr['pixdim'][1:3] 

        # field of view in mm
        FOV = np.multiply(voxel_size,volume_size_1)
    else:
        dset = ismrmrd.Dataset(filename, 'dataset', create_if_needed=False)
        header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
        enc = header.encoding[0]

        eFOVx = enc.encodedSpace.fieldOfView_mm.x
        eFOVy = enc.encodedSpace.fieldOfView_mm.y
        eFOVz = enc.encodedSpace.fieldOfView_mm.z   

        if(acq == '3D'):
            # field of view in mm
            FOV = np.array((eFOVx, eFOVy, eFOVz))
        else:
            FOV = np.array((eFOVx, eFOVy))

    
    if(acq == '3D'):
        volume_size = IM_3D_unseq.shape[1:]
    else:
        #matrix size 
        volume_size = IM_3D_unseq.shape[2:]
    
    # kspace spacing in 1/m
    delta_k = 1000/FOV

    

    #set the maximum radius and width
    r_max = int(np.sqrt(np.square(np.multiply(np.multiply(volume_size,0.5),delta_k)).sum()))
    w_max = int(0.1*r_max)
    width = 10
    print(f'R_MAX {r_max}')
    
    if ( acq == '3D'):
        IM_3D = np.expand_dims(IM_3D_unseq, axis=0)
    else:
        IM_3D = IM_3D_unseq

    im_norm = np.zeros(IM_3D.shape) 
    ncoils = IM_3D.shape[1]
    nslices = IM_3D.shape[0] 
    energy_norm_coef = np.zeros((nslices,ncoils,3)) #save energy coef of norm image

    for slice in range(nslices):
        for coil in range(ncoils):
            im_i = IM_3D[slice,coil,...]
            #saving energy fitting coefficient
            xdata = []
            ydata = []

            #centre of the disk
            k0 = np.unravel_index(np.argmax(im_i, axis=None), im_i.shape)
            #print(k0)
            
            kx = np.multiply((np.array(range(0,volume_size[0]))-k0[0]),delta_k[0])
            ky = np.multiply((np.array(range(0,volume_size[1]))-k0[1]),delta_k[1])

            if ( acq == '3D'):
                kz = np.multiply((np.array(range(0,volume_size[2]))-k0[2]),delta_k[2])
                #np.mgrid and np.meshgrid() do the same thing but the first and the second axis are swapped,kx & ky is swapped
                KX,KY,KZ = np.meshgrid(ky,kx,kz)
                KR = np.sqrt(KX*KX + KY*KY + KZ*KZ)
            else:
                KX,KY = np.meshgrid(ky,kx)
                KR = np.sqrt(KX*KX + KY*KY)

            for kr in range(0,r_max,width):
                if ( acq == '3D'):
                    array = np.zeros(volume_size)
                    #get the donuts mask 
                    mask_outer = KX*KX + KY*KY + KZ*KZ  <= (kr + width)*(kr+width)
                    mask_inner =  KX*KX + KY*KY + KZ*KZ >= kr*kr
                #2D
                else:
                    array = np.zeros((IM_3D.shape[2:]))
                    #get the donuts mask 
                    mask_outer = KX*KX + KY*KY  <= (kr + width)*(kr+width)
                    mask_inner =  KX*KX + KY*KY >= kr*kr
            
                #Shell binray mask
                mask = np.multiply(mask_outer,mask_inner)

                array[mask] = 1

                #get energy normalization coefficient
                roi = np.multiply(im_i,array)
                num = np.count_nonzero(array==1)

                if (num>0):
                    energy = (np.linalg.norm(roi)**2) / num
                else:
                    break
                xdata.append(kr)
                ydata.append(energy)

            #print(xdata,ydata)
            #curve fitting y = a*e^-(b*x) + c
            popt, pcov = curve_fit(funcEnergy, xdata, ydata)
            #print(f'slice {slice} coil {coil} {popt}')
            coef = funcEnergy(KR, *popt)
            im_norm[slice,coil] = 1/coef
            #print(im_norm[slice,coil])
            energy_norm_coef[slice,coil] = [*popt]
    
    #change to complex
    im_norm_complex = im_norm + 1j*im_norm
    
    if ( acq == '3D'): 
        return im_norm_complex[0]
    else:
        return im_norm_complex 

#f(x) = a*exp(-bx) + c a,b,c are hyperparameter to be choosen manually 
def funcExponential(x, a, b, c, n=1):
    return a * np.exp(-1*np.power(x/b,n)) + c

#https://www.sciencedirect.com/science/article/pii/S1053811914006922?via%3Dihub#f0015 => 
def fucnHighPass(x,K0):
    return np.power(x,4)/(np.power(x,4) + np.power(K0,4))

def getWeightedEnergyCoef(filename,IM_3D_unseq,mse_type='weighted',acq='2D',K0=1,noise_level=1,n=1):
    extension = os.path.splitext(filename)[1]

    if(extension == '.nii'):
        
        img = ni.load(filename)
        hdr = img.header
        
        if(acq == '3D'):
            # matrix size
            volume_size_1 = hdr['dim'][1:4]

            # voxel dimension
            voxel_size = hdr['pixdim'][1:4]
        else:
            # matrix size
            volume_size_1 = hdr['dim'][2:4]

            # voxel dimension
            voxel_size = hdr['pixdim'][1:3]

        # field of view in mm
        FOV = np.multiply(voxel_size,volume_size_1)
        if(acq == '3D'):
            
            # field of view in mm
            volume_size = IM_3D_unseq.shape[1:]
        else:
  
            #matrix size 
            volume_size = IM_3D_unseq.shape[2:]
    else:
        dset = ismrmrd.Dataset(filename, 'dataset', create_if_needed=False)
        header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
        enc = header.encoding[0]

        eFOVx = enc.encodedSpace.fieldOfView_mm.x
        eFOVy = enc.encodedSpace.fieldOfView_mm.y
        eFOVz = enc.encodedSpace.fieldOfView_mm.z   

        if(acq == '3D'):
            
            # field of view in mm
            FOV = np.array((eFOVx, eFOVy, eFOVz))
            volume_size = IM_3D_unseq.shape[1:]
        else:
  
            FOV = np.array((eFOVx, eFOVy))
            #matrix size 
            volume_size = IM_3D_unseq.shape[2:]

    
    # kspace spacing in 1/m
    delta_k = 1000/FOV

    #set the maximum radius and width
    r_max = int(np.sqrt(np.square(np.multiply(np.multiply(volume_size,0.5),delta_k)).sum()))
    width = 10
    print(f'R_MAX {r_max}')


    if ( acq == '3D'):
        IM_3D = np.expand_dims(IM_3D_unseq, axis=0)
    else:
        IM_3D = IM_3D_unseq 

    im_norm = np.zeros(IM_3D.shape)
    ncoils = IM_3D.shape[1]
    nslices = IM_3D.shape[0] 

    for slice in range(nslices):
        for coil in range(ncoils):
            
            im_i = IM_3D[slice,coil,...]
            #centre of the disk
            k0 = np.unravel_index(np.argmax(im_i, axis=None), im_i.shape)
            #print(k0)
            
            kx = np.multiply((np.array(range(0,volume_size[0]))-k0[0]),delta_k[0])
            ky = np.multiply((np.array(range(0,volume_size[1]))-k0[1]),delta_k[1])

            if ( acq == '3D'):
                kz = np.multiply((np.array(range(0,volume_size[2]))-k0[2]),delta_k[2])
                #np.mgrid and np.meshgrid() do the same thing but the first and the second axis are swapped,kx & ky is swapped
                KX,KY,KZ = np.meshgrid(ky,kx,kz)
                KR = np.sqrt(KX*KX + KY*KY + KZ*KZ)
            else:
                KX,KY = np.meshgrid(ky,kx)
                KR = np.sqrt(KX*KX + KY*KY)

            #fit into function f(x) = a*exp(-bx) + c 
            #a = max(signal) & c = noise levels & b is cutoff frequency 

            if(mse_type=='weighted'):
                popt = (np.max(np.abs(im_i))**n, K0, noise_level*np.min(np.abs(im_i))**n)
                im_norm[slice,coil] = 1/funcExponential(KR, *popt,n=n)
            else:
                im_norm[slice,coil] = fucnHighPass(KR,K0)

    #change to complex
    im_norm_complex = im_norm + 1j*im_norm
    if ( acq == '3D'): 
        return im_norm_complex[0]
    else:
        return im_norm_complex 


#high pass filtering by distance
def highPassByDistance(filename,IM_3D_unseq,acq='2D', kr=10):
    extension = os.path.splitext(filename)[1]

    if(extension == '.nii'):
        
        img = ni.load(filename)
        hdr = img.header
        
        if(acq == '3D'):
            IM_3D = np.expand_dims(IM_3D_unseq, axis=0)
            # matrix size
            volume_size = hdr['dim'][1:4]

            # voxel dimension
            voxel_size = hdr['pixdim'][1:4]
        else:
            IM_3D = IM_3D_unseq 
            # matrix size
            volume_size = IM_3D.shape[2:]

            # voxel dimension
            voxel_size = hdr['pixdim'][1:3] 

        # field of view in mm
        FOV = np.multiply(voxel_size,volume_size)
    else:
        dset = ismrmrd.Dataset(filename, 'dataset', create_if_needed=False)
        header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
        enc = header.encoding[0]

        eFOVx = enc.encodedSpace.fieldOfView_mm.x
        eFOVy = enc.encodedSpace.fieldOfView_mm.y
        eFOVz = enc.encodedSpace.fieldOfView_mm.z   

        if(acq == '3D'):
            #add a slice dimension
            IM_3D = np.expand_dims(IM_3D_unseq, axis=0)
            
            # field of view in mm
            FOV = np.array((eFOVx, eFOVy, eFOVz))
            volume_size = IM_3D.shape[1:]
        else:
            
            IM_3D = IM_3D_unseq 
            FOV = np.array((eFOVx, eFOVy))

            #matrix size 
            volume_size = IM_3D.shape[2:]

    
    # kspace spacing in 1/m
    delta_k = 1000/FOV

    #centre of the disk
    k0 = np.multiply(volume_size, 0.5).astype('int')

    kx = np.multiply((np.array(range(0,volume_size[0]))-k0[0]),delta_k[0])
    ky = np.multiply((np.array(range(0,volume_size[1]))-k0[1]),delta_k[1])

    #set the maximum radius and width
    r_max = int(np.sqrt(np.square(np.multiply(np.multiply(volume_size,0.5),delta_k)).sum()))
    w_max = int(0.1*r_max)
    print(f'R_MAX {r_max}')

    array = np.zeros(volume_size).astype('complex64')

    if ( acq == '3D'):
        im_high_pass = np.zeros(IM_3D.shape).astype('complex64')
        kz = np.multiply((np.array(range(0,volume_size[2]))-k0[2]),delta_k[2])
        #np.mgrid and np.meshgrid() do the same thing but the first and the second axis are swapped,kx & ky is swapped
        KX,KY,KZ = np.meshgrid(ky,kx,kz)
        KR = np.sqrt(KX*KX + KY*KY + KZ*KZ)
        mask = KX*KX + KY*KY + KZ*KZ  <= kr*kr

    else:
        im_high_pass = np.zeros(IM_3D.shape).astype('complex64')
        KX,KY = np.meshgrid(ky,kx)
        KR = np.sqrt(KX*KX + KY*KY)
        mask = KX*KX + KY*KY  <= kr*kr

    array[~mask] = 1
    
    im_high_pass[:,:,...]  = np.multiply(IM_3D[:,:,...],array)
    return im_high_pass,array