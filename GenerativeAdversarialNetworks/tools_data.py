# -*- coding: utf-8 -*-
from tools_general import np, rng
import os, sys
from tools_config import *

import glob
import cv2

def retransform(images):
    return ((images +1) * 127.5).astype(np.uint8)

def transform(images):
    return (images / 127.5) - 1.

def jpg2array(inImg_dir, inLabel_dir, do_augment = False):
    # 1: get list of all images 
    in_images = list(os.listdir(inImg_dir))
    datas = []
    
    in_images = [ inImg_dir + img for img in  in_images if '.jpg' in img]

    # 2: loop over all related csv files, for the specified timeframe and symbol    
    for inImg in in_images:
    # 3.1: extract data to an array
        im_name = os.path.basename(inImg)
        
        im = cv2.imread(inImg)
        label = cv2.imread(inLabel_dir + im_name.replace('A', 'B'))
        
        im_flipped = cv2.flip(im,1)
        label_flipped = cv2.flip(label,1)
        
        im = im[np.newaxis, ...]
        label = label[np.newaxis, ...]

        data = np.concatenate([im, label], axis=3)
        
        if do_augment:
            
            im_flipped = im_flipped[np.newaxis, ...]
            label_flipped = label_flipped[np.newaxis, ...]

            data_flipped = np.concatenate([im_flipped, label_flipped], axis=3)
        
            data = np.concatenate([data, data_flipped], axis=0)

    # 3.3: append data to a common array    
        if len(datas) == 0:
            datas = data
        else:
            datas = np.concatenate([datas, data], axis=0)
    
    # 4: return the array
    return datas
         
def create_cmp_db(in_dir):  
    networktype = 'img2imgGAN_CMP'
    
    work_dir = in_dir + networktype + '/'

    train_inImg_dir = work_dir + 'trainA/'
    train_inLabel_dir = work_dir + 'trainB/'
        
    test_inImg_dir = work_dir + 'testA/'
    test_inLabel_dir = work_dir + 'testB/'
    
    out_dir = work_dir 
    
    train_data = jpg2array(train_inImg_dir, train_inLabel_dir, do_augment = True)
    test_data = jpg2array(test_inImg_dir, test_inLabel_dir, do_augment = False)
    
    train_data = transform(train_data)
    test_data = transform(test_data)
       
    Xmean = train_data.mean(axis=1).mean(axis=1).mean(axis=0)[np.newaxis, np.newaxis, np.newaxis, :]
    Xstd = np.std(train_data, axis=1).std(axis=1).std(axis=0)[np.newaxis, np.newaxis, np.newaxis, :]

    np.savez_compressed(out_dir + 'mean_std.npz', Xmean=Xmean, Xstd=Xstd)
    
    np.savez(out_dir + 'eval', data=test_data[0:10])    
    np.savez(out_dir + 'test', data=test_data)
    np.savez(out_dir + 'train', data=train_data)
       

if __name__ == '__main__': 
    create_cmp_db(data_dir)