import sys
sys.path.insert(0, './')

import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import time
import argparse
import random
import uuid
import numpy as np
import pandas as pd
from copy import deepcopy
import glob
import json
from PIL import Image
try:
    from tqdm import tqdm
    tqdm = tqdm
except:
    print("can't import tqdm. progress bar is disabled")
    tqdm = lambda x: x
from torchvision.datasets.folder import default_loader as img_loader
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import skimage
from PIL import Image, ImageEnhance
from tensorflow.keras.preprocessing import image
from heapq import nlargest, nsmallest
from operator import itemgetter

from skimage.metrics import mean_squared_error

from biggan_generator import *



def tile_permutations(img_one, img_two):
    half_height = img_one.shape[2]//2
    half_width = img_one.shape[3]//2
    
    merged_images = []
    
    top_halves = [img_one[:,:,:half_height, :], img_two[:,:,:half_height, :]]
    bottom_halves = [img_one[:,:,half_height:, :], img_two[:,:,half_height:, :]]
    
    for i, top in enumerate(top_halves):
        for j, bottom in enumerate(bottom_halves):
            if i!=j:
                merged = np.concatenate([top, bottom], axis = 2)
                merged_images.append(merged)
    
    top_lefts = [img_one[:,:,:half_height, :half_width], img_two[:,:,:half_height, :half_width]]
    top_rights = [img_one[:,:,:half_height, half_width:], img_two[:,:,:half_height, half_width:]]
    bottom_lefts = [img_one[:,:,half_height:, :half_width], img_two[:,:,half_height:, :half_width]]
    bottom_rights = [img_one[:,:,half_height:, half_width:], img_two[:,:,half_height:, half_width:]]
    
    for i, top_left in enumerate(top_lefts):
        for j, top_right in enumerate(top_rights):
            for k, bottom_left in enumerate(bottom_lefts):
                for l, bottom_right in enumerate(bottom_rights):
                    if not(i==j and j==k and k==l):
                        top = np.concatenate([top_left, top_right], axis = 3)
                        bottom = np.concatenate([bottom_left, bottom_right], axis = 3)
                        merged = np.concatenate([top, bottom], axis = 2)
                        merged_images.append(merged)
    
    return merged_images



def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def save_img(img_tensor, img_path, img_label, save_path):
    img_arr = img_tensor.mul_(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy() 
    im = Image.fromarray(img_arr)
    im.save(save_path)

def setup_df(dataset_root):
    table = []
    
    paths_good = glob.glob(os.path.join(dataset_root,"19-01 goed/*.png"))
    paths_double_print = glob.glob(os.path.join(dataset_root,"19-01 dubbeldruk/*.png"))
    paths_interrupted = glob.glob(os.path.join(dataset_root,"19-01 onderbroken/*.png"))
    
    all_paths = paths_good + paths_interrupted + paths_double_print

    for path in all_paths:
        label = path.split("/")[-2]
        table.append({"label":label,"path":path,"img_name":path.split("/")[-1]})
    
    df_dataset = pd.DataFrame(table)
    df_dataset = df_dataset.sort_values(["path"])
        
    return df_dataset

def get_instance_number(class_label, dataset_root):
    path = os.path.join(dataset_root, class_label)
    return len(os.listdir(path))

def determine_save_gen_number(AUGMENTATION_TARGET, base_set_size, dataset_root, coeff = 1.0):
    numbers = {}

    target_number = coeff*AUGMENTATION_TARGET

    for label in ['19-01 goed', '19-01 dubbeldruk', '19-01 onderbroken']:
        actual_number = get_instance_number(label, dataset_root)
        if target_number > actual_number:
            numbers[label] = int(round((target_number - base_set_size) / base_set_size))
        else:
            numbers[label] = 0    
    return numbers



def generate_images(dataset_root, weights_path, AUGMENTATION_TARGET, savedir, no_images_to_generate, GLOBAL_SEED,
    start_image = 0, filter_set = None):
    df_dict =  setup_df(dataset_root)    
    dataset_df = df_dict.sort_values("path")

    transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    device = try_gpu()
    G = Generator().to(device)
    G.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    
    model_ori = AdaBIGGAN(G,dataset_size=1)
    model_ori.eval()

    criterion = AdaBIGGANLoss(
        scale_per=0.1,
        scale_emd=0.1,
        scale_reg=0,
        normalize_img = 0,
        normalize_per = 0,
        dist_per = "l2",
        dist_img = "l1",
    )
    model_ori = model_ori.to(device)
    criterion = criterion.to(device)
    indices = torch.LongTensor([0]).to(device)
    
    max_iter = 500

    save_gen = determine_save_gen_number(AUGMENTATION_TARGET, len(filter_set), dataset_root)
    num_gen = 4*max(save_gen.values())

    truc = 0.4
    
    dataset_df = pd.concat([dataset_df[(dataset_df['label'] == k) & (v > 0)] for k,v in save_gen.items()])
    if filter_set is not None:
        dataset_df = pd.concat([dataset_df[(dataset_df['img_name'] == n)] for n in filter_set])
    no_images_to_generate = min(no_images_to_generate, len(dataset_df))
    for i in tqdm(range(start_image, no_images_to_generate)):     
        np.random.seed(GLOBAL_SEED)
        torch.manual_seed(GLOBAL_SEED)
        random.seed(GLOBAL_SEED)
        model = deepcopy(model_ori)
        optimizer,scheduler = setup_optimizer(model,
                    lr_g_batch_stat=0.0005,
                    lr_g_linear=0,
                    lr_bsa_linear=0.0005,
                    lr_embed=0.01,
                    lr_class_cond_embed=0.03,
                    step=500,
                    step_facter=0.1)

        img_path = dataset_df.iloc[i]["path"]
        img_label = dataset_df.iloc[i]["label"]
        img_name = dataset_df.iloc[i]["img_name"].split(".")[0]
        
        
        img_save_dir = os.path.join(savedir, img_label)
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
   
        img_loaded = img_loader(img_path)
        img = transform(img_loaded).unsqueeze(0)
        img = img.to(device)

        model.eval()

        start = time.time()
        for iteration in range(max_iter):
            scheduler.step()

            embeddings = model.embeddings(indices)
            embeddings_eps = torch.randn(embeddings.size(),device=device)*0.05
            embeddings += embeddings_eps 

            img_generated = model(embeddings)
            loss = criterion(img_generated,img,embeddings,model.linear.weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        np.random.seed(GLOBAL_SEED)
        torch.manual_seed(GLOBAL_SEED)
        random.seed(GLOBAL_SEED)

        with torch.no_grad():
            embeddings = model.embeddings(indices)
            embeddings = embeddings*torch.randint(2,size=(num_gen,120),dtype=embeddings.dtype,device=device)
            embeddings_eps = torch.randn((num_gen,120),device=device)*0.2
            embeddings +=embeddings_eps 
            embeddings = torch.clamp(embeddings,-truc,truc)
            
            i = 0
            saved = 0 
            
            synthetic_images_with_losses = []
            for i in range(num_gen):
                img_generated = model(embeddings[i].unsqueeze(0))
                img_generated_normalized = img_generated.add_(1.0).div_(2).to('cpu').numpy() 

                resize_transform = transforms.Compose([
                    transforms.Resize((128, 128))
                ])
                img_original_normalized = resize_transform(img.cpu())
                
                hybrid_images = tile_permutations(img_generated_normalized,img_original_normalized)
                
                for h_img in hybrid_images:
                    loss = mean_squared_error(img.cpu().numpy()[0], h_img[0])
                    synthetic_images_with_losses.append((h_img, loss))
                    
            best_gen_images_w_loss = nsmallest(save_gen[img_label], synthetic_images_with_losses, key=itemgetter(1))
            
            for i, tup in enumerate(best_gen_images_w_loss):
                img_gen, l = tup
                img_gen = torch.from_numpy(img_gen[0])
                    
                save_path = os.path.join(img_save_dir, "%s_gen%d_loss_%0.4f.png"%(img_name,i,l))
                save_img(img_gen, img_path, img_label, save_path)

        elapsed_time = time.time() - start
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")