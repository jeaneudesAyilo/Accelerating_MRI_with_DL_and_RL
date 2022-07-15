#!/usr/bin/env python
# coding: utf-8

import numpy as np
import nibabel as ni
import os
import math

verbose = True


def save_nifty(array_np,filename):
    new_image = ni.Nifti1Image(array_np, affine=np.eye(4))
    t=ni.save(new_image, filename)
    return t

def load_nifty(filename):
    img_nifty = ni.load(filename)
    img_array = np.array(img_nifty.get_fdata())
    return img_array

def save_complex_nifty(array_np,filename,axis=3):
    s = list(array_np.shape)
    s = s[0:axis] + [1] + s[axis:] 
    array_npc = np.reshape(array_np,s)
    array_npc = np.concatenate((np.real(array_npc),np.imag(array_npc)),axis=axis)
    print(f"array_save_nifty{array_npc.shape}")
    t = save_nifty(array_npc,filename)
    return t

def load_complex_nifty(filename,axis=3):
    img_array = load_nifty(filename)
    s = list(img_array.shape)
    indicesr=range(0,s[axis],2)
    indicesi=range(1,s[axis],2)
    img_arrayc = np.take(img_array, indicesr, axis=axis) +1j*np.take(img_array, indicesi, axis=axis) 
    return img_arrayc