#!/usr/bin/env python
# coding: utf-8

    ##here i am generating different contrast of brain. To each of them, I add a different noise
import numpy as np
import matplotlib.pyplot as plot
import nibabel as ni
import os
import math
from utils.load_data import save_nifty,load_nifty, save_complex_nifty,save_complex_nifty, load_complex_nifty 
from utils.misc import add_noise

verbose = True
np.set_printoptions(formatter={'float_kind':"{:.2f}".format})

if __name__ == "__main__":
    # creating fake image with a cylinder
    #nx = 256
    #ny = 256
    #nz = 160
    #d = 0.4 # cylinder diameter
    #l = 0.75 # cylinder length
    #x = np.linspace(-0.5, 0.5, nx)
    #y = np.linspace(-0.5, 0.5, ny)
    #z = np.linspace(-0.5, 0.5, nz)
    #x, y, z= np.meshgrid(x, y, z)
    #cylinder = (1*(np.sqrt(x*x+y*y)<d/2))*(1*(np.abs(z)<l/2))

    #if verbose:
    #    print('Writing cylinder as nifti...')
    #save_nifty(cylinder,'./data/output/cylinder.nii')

    #image = cylinder

    tpm='./data/TPM/TPM.nii'
    num_images = 1
    if  os.path.isfile(tpm):
        if verbose:
            print('Loading template...')
        data = load_nifty(tpm) # this is a template
        print(f"{data.shape}")

    for i in range(num_images):
        # generating fake 3D image
        if verbose:
            print('Generating image ...')
        #contrast =[white_matter, grey_matter, csf, skull, brain_bound, other ]    
        #contrast = [0.2,0.8,0.1,0,0.2,0]
        #contrasts = np.array([[0.8,1.0,0.4,0,1.0,0]])
        #contrast = np.around(20-20*np.random.random(6),decimals=1)
        
        np.random.seed(123)
        contrasts = np.around(20-20*np.random.random((1,6)),decimals=1)
        contrasts[:,(3,5)]=0
        
        np.random.seed(123)
        seed_list = np.random.choice(1000, size=contrasts.shape[0], replace=False)
        
        
        ##generate different contrast images
        for i in range(contrasts.shape[0]):
            
            print(f"contrast {contrasts[i]}")

            #data_dir = f'F:/data_for_mask_learning/output/c_wm_{contrasts[i][0]}_gm_{contrasts[i][1]}_csf_{contrasts[i][2]}_br_{contrasts[i][4]}'
                       
            
            #data_dir = f'E:/data_for_mask_learning/output/c_wm_{contrasts[i][0]}_gm_{contrasts[i][1]}_csf_{contrasts[i][2]}_br_{contrasts[i][4]}'
            
        #data_dir = f"F:/data/output/c_wm_{contrast[0]}_gm_{contrast[1]}_csf_{contrast[2]}_br_{contrast[4]}"
        
            #print(f'[INFO] Creating directory {data_dir} to store output.')
            #os.mkdir(data_dir)
        
        
            #data_dir_noise = f"F:/data_for_mask_learning/output/c_wm_{contrasts[i][0]}_gm_{contrasts[i][1]}_csf_{contrasts[i][2]}_br_{contrasts[i][4]}_noise_s{seed_list[i]}"
            
            #data_dir_noise = f"/data1/home/jean-eudes.ayilo/Pytorch-MRI-ML-recon-V1.1/data_for_mask_learning/sensitivity/c_wm_{contrasts[i][0]}_gm_{contrasts[i][1]}_csf_{contrasts[i][2]}_br_{contrasts[i][4]}_noise_s{seed_list[i]}"

            #os.mkdir(data_dir_noise) 


            image=0
            for c in range(data.shape[3]):
                image += contrasts[i][c]*data[:,:,:,c]
            #save_nifty(image,f'{data_dir}/image.nii')
            #data = data[:,:,:,0]+data[:,:,:,]

            # generating coil sensitivity
            s = image.shape
            radius = 1 # coil will be located on a circle
            ncoils = 8 # number of coils
            # modeled as a point dipole
            x = np.linspace(-s[0]/2,s[0]/2, s[0])
            y = np.linspace(-s[1]/2, s[1]/2, s[1])
            z = np.linspace(-s[2]/2, s[2]/2, s[2])
            y, x, z= np.meshgrid(y, x, z)

            sb = list(s)
            sb+=[2*ncoils]
            sc = list(s)
            sc+=[ncoils]
            B = np.zeros(sb)
            for coil in range(ncoils):
                phi = 2*math.pi*coil/ncoils
                # print(phi)
                posx = radius*s[0]*math.cos(phi)
                posy = radius*s[1]*math.sin(phi)
                X = x-posx
                Y = y-posy
                X0=X
                X = X*math.cos(phi)+Y*math.sin(phi)
                Y = -X0*math.sin(phi)+Y*math.cos(phi)
                Z = z
                thetay = X/np.sqrt(X*X+Y*Y+Z*Z)
                thetax = Y/np.sqrt(X*X+Y*Y+Z*Z)
                #phi = Y/Z
                # real part
                B[:,:,:,2*coil] = (3*thetay*thetay-1)/(np.sqrt(X*X+Y*Y+Z*Z)**(3))
                # imag part
                B[:,:,:,2*coil+1] = (3*thetay*thetax)/(np.sqrt(X*X+Y*Y+Z*Z)**(3))
            save_nifty(B,'./data/output/sensitivity.nii')

print("###########  FIN  ###########")            
