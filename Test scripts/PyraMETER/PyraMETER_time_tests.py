########################################################## Globals ##########################################################

global_var = {
    # Resolutions
    'RGB_img_res': (3, 192, 256),
    'D_img_res': (1, 48, 64),
    # Operations
    'do_prints': True,
    'do_print_model': True,
    'do_pretrained': True,
    'do_train': False,
    'do_print_best_worst': True,
    # ImageNet Initialization
    'imagenet_w_init': False,
    'imagenet_enc': 'METER_nyu_encoder_aug_mean0_var1_59', # A voi non interessa
    # Parameters
    'dts_type': 'nyu',
    'architecture_type': 's',
    'seed': 10000,
    'lr': 1e-3,
    'lr_patience': 15,
    'epochs': 80, # Scegliete VOI,
    'batch_size': 64,
    'batch_size_eval': 1,
    'n_workers': 2,
    'e_stop_epochs': 30,
    'size_train': None,
    'size_test': None,
}

augmentation_parameters = {
    'flip': 0.5,
    'mirror': 0.5,
    'color&bright': 0.5,
    'c_swap': 0.5,
    'random_crop': 0.5,
    'random_d_shift': 0.5  # range(+-10)cm
}

dataset_root = '../'
save_model_root = './models/'
imagenet_init = '/work/imagenet/' # Non serve, ma da tenere

########################################################## Imports ##########################################################

import shutil
import pandas as pd
import numpy as np
import os
import skimage.transform as st
import torch
import pickle
import matplotlib.pyplot as plt
from torchsummaryX import summary
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm
import gc
import random
import torchvision.transforms as TT
import torchvision.transforms.functional as TF
from PIL import Image
from itertools import product
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from einops import rearrange
import csv
import math
from time import perf_counter
import matplotlib.pyplot as plt
import sys
import warnings
import argparse

network_type = "PyraMETER"
old_stout = sys.stdout

########################################################## Utils ##########################################################

def hardware_check():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Actual device: ", device)
    if 'cuda' in device:
        print("Device info: {}".format(str(torch.cuda.get_device_properties(device)).split("(")[1])[:-1])

    return device


def plot_depth_map(dm):

    MIN_DEPTH = 0.0
    MAX_DEPTH = min(np.max(dm.numpy()), np.percentile(dm, 99))

    dm = np.clip(dm, MIN_DEPTH, MAX_DEPTH)
    cmap = plt.cm.plasma_r

    return dm, cmap, MIN_DEPTH, MAX_DEPTH


def resize_keeping_aspect_ratio(img, base):
    """
    Resize the image to a defined length manteining its proportions
    Scaling the shortest side of the image to a fixed 'base' length'
    """

    if img.shape[0] <= img.shape[1]:
        basewidth = int(base)
        wpercent = (basewidth / float(img.shape[0]))
        hsize = int((float(img.shape[1]) * float(wpercent)))
        img = st.resize(img, (basewidth, hsize), anti_aliasing=False, preserve_range=True)
    else:
        baseheight = int(base)
        wpercent = (baseheight / float(img.shape[1]))
        wsize = int((float(img.shape[0]) * float(wpercent)))
        img = st.resize(img, (wsize, baseheight), anti_aliasing=False, preserve_range=True)

    return img


def compute_rmse(predictions, depths):
    valid_mask = depths > 0.0
    valid_predictions = predictions[valid_mask]
    valid_depths = depths[valid_mask]
    mse = (torch.pow((valid_predictions - valid_depths).abs(), 2)).mean()
    return torch.sqrt(mse)


def compute_accuracy(y_pred, y_true, thr=0.05):
    valid_mask = y_true > 0.0
    valid_pred = y_pred[valid_mask]
    valid_true = y_true[valid_mask]
    correct = torch.max((valid_true / valid_pred), (valid_pred / valid_true)) < (1 + thr)
    return 100 * torch.mean(correct.float())


def print_model(model, device, save_model_root, input_shape):
    info = summary(model, torch.ones((1, input_shape[0], input_shape[1], input_shape[2])).to(device))
    info.to_csv(save_model_root + 'model_summary.csv')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(model, name, path_save_model):
    """
    Saves a model
    """
    if '_best' in name:
        folder = name.split("_best")[0]
    elif '_checkpoint' in name:
        folder = name.split("_checkpoint")[0]
    if not os.path.isdir(path_save_model):
        os.makedirs(path_save_model, exist_ok=True)
    torch.save(model.state_dict(), path_save_model + name)


def save_history(history, filepath):
    tmp_file = open(filepath + '.pkl', "wb")
    pickle.dump(history, tmp_file)
    tmp_file.close()


def save_csv_history(model_name, path):
    objects = []
    with (open(path + model_name + '_history.pkl', "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    df = pd.DataFrame(objects)
    df.to_csv(path + model_name + '_history.csv', header=False, index=False, sep=" ")


def load_pretrained_model(model, path_weigths, device, do_pretrained, imagenet_w_init):
    model_name = model.__class__.__name__

    if do_pretrained:
        print("\nloading checkpoint for entire {}..\n".format(model_name))
        model_dict = torch.load(path_weigths, map_location=torch.device(device))
        model.load_state_dict(model_dict)
        print("checkpoint loaded\n")

    if imagenet_w_init:
        print("\nloading checkpoint from ImageNet {}..\n".format(model_name))
        pretrained_dict = torch.load(path_weigths, map_location=torch.device(device))
        model_dict = model.state_dict()
        print('Pretained on ImageNet has: {} trainable parameters'.format(len(pretrained_dict.items())))

        # pretrained_param = len(pretrained_dict.items())
        counter_param = 0
        for i, j in pretrained_dict.items():
            if (i in model_dict) and model_dict[i].shape == pretrained_dict[i].shape:
                counter_param += 1

        print(f'Pertained parameters: {counter_param}\n')

        # 1. filter out unnecessary keys
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        # alternativa to 2 e 3
        # model.load_state_dict(pretrained_dict, strict=False)
        print("Partial initialization computed\n")

    return model, model_name


def plot_graph(f, g, f_label, g_label, title, path):
    epochs = range(0, len(f))
    plt.plot(epochs, f, 'b', label=f_label)
    plt.plot(epochs, g, 'orange', label=g_label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid('on', color='#cfcfcf')
    plt.tight_layout()
    plt.savefig(path + title + '.pdf')
    plt.close()


def plot_history(history, path):
    plot_graph(history['train_loss'], history['val_loss'], 'Train Loss', 'Val. Loss', 'TrainVal_loss', path)
    plot_graph(history['train_acc'], history['val_acc'], 'Train Acc.', 'Val. Acc.', 'TrainVal_acc', path)


def plot_loss_parts(history, path, title):
    l_mae_list = history['l_mae']
    l_norm_list = history['l_norm']
    l_grad_list = history['l_grad']
    l_ssim_list = history['l_ssim']
    epochs = range(0, len(l_mae_list))
    plt.plot(epochs, l_mae_list, 'r', label='l_mae')
    plt.plot(epochs, l_norm_list, 'g', label='l_norm')
    plt.plot(epochs, l_grad_list, 'b', label='l_grad')
    plt.plot(epochs, l_ssim_list, 'orange', label='l_ssim')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.grid('on', color='#cfcfcf')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + title + '.pdf')
    plt.close()


def print_img(dataset, label, save_model_root, index=None, quantity=1, print_info_aug=False):
    for i in range(quantity):
        img, depth = dataset.__getitem__(index, print_info_aug)

        print(f'Depth -> Shape = {depth.shape}, max = {torch.max(depth)}, min = {torch.min(depth)}')
        print(f'IMG -> Shape = {img.shape}, max = {torch.max(img)}, min = {torch.min(img)}, mean = {torch.mean(img)},'
              f' variance =  {torch.var(img)}\n')

        fig = plt.figure(figsize=(15, 3)) # 15 NYU # 30 KITTI
        plt.subplot(1, 3, 1)
        plt.title('Input image')
        plt.imshow(torch.moveaxis(img, 0, -1), cmap='gray', vmin=0.0, vmax=1.0)
        if not False:
            plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Grayscale DepthMap')
        plt.imshow(torch.moveaxis(depth, 0, -1), cmap='gray', interpolation='nearest')
        plt.colorbar()
        if not False:
            plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Colored DepthMap')
        depth, cmap_dm, vmin, vmax = plot_depth_map(depth)
        plt.imshow(torch.moveaxis(depth, 0, -1), cmap=cmap_dm, vmin=vmin, vmax=vmax, interpolation='nearest')
        plt.colorbar()
        if not False:
            plt.axis('off')

        print("************************** ",save_model_root)
        save_path = save_model_root + 'example&augment_img/'
        print("************************** ",save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.tight_layout()
        plt.savefig(save_path + 'img_' + str(i) + '_' + label + '.pdf')
        plt.close(fig=fig)


def save_prediction_examples(model, dataset, device, indices, save_path, ep):
    """
    Shows prediction example
    """
    fig = plt.figure(figsize=(20, 3)) # 20 NYU # 40 KITTI
    for i, index in zip(range(len(indices)), indices):
        img, depth = dataset.__getitem__(index)
        img = np.expand_dims(img, axis=0)
        # Predict
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(img).to(device))
            # Build plot
            _, cmap_dm, vmin, vmax = plot_depth_map(depth)
            plt.subplot(1, len(indices), i+1)
            plt.imshow(np.squeeze(pred.cpu()), cmap=cmap_dm, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar()
            cbar.ax.set_xlabel('cm', size=13, rotation=0)
            if False:
                plt.axis('off')

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.tight_layout()
    plt.savefig(save_path + 'img_ep_' + str(ep) + '.pdf')
    plt.close(fig=fig)


def save_best_worst(list_type, type, model, dataset, device, save_model_root):
    save_path = save_model_root + type + '_predictions/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(len(list_type)):
        index_image = list_type[i][0]
        rmse_value = list_type[i][1]

        img, depth = dataset.__getitem__(index=index_image)

        fig = plt.figure(figsize=(18, 3)) # 18 NYU # 40 KITTI
        plt.subplot(1, 4, 1)
        plt.title(f'Original image {index_image}')
        plt.imshow(torch.moveaxis(img, 0, -1), cmap='gray', vmin=0.0, vmax=1.0)
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title('Ground Truth')
        depth, cmap_dm, vmin, vmax = plot_depth_map(depth)
        plt.imshow(torch.moveaxis(depth, 0, -1), cmap=cmap_dm, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.axis('off')

        # Predict
        model.eval()
        with torch.no_grad():
            pred = model(torch.unsqueeze(img, dim=0).to(device))

        plt.subplot(1, 4, 3)
        plt.title('Predicted DepthMap')
        pred, cmap_dm, _, _ = plot_depth_map(torch.squeeze(pred.cpu(), dim=0))
        plt.imshow(torch.moveaxis(pred, 0, -1), cmap=cmap_dm, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title('Disparity Map, RMSE = {:.2f}'.format(rmse_value))
        intensity_img = torch.moveaxis(torch.abs(depth - pred), 0, -1)
        plt.imshow(intensity_img, cmap=plt.cm.magma, vmin=0)
        plt.colorbar()
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path + '/seq_' + str(i) + '.pdf')
        plt.close(fig=fig)


def compute_MeanVar(dataset):
    r_mean, g_mean, b_mean = [], [], []
    r_var, g_var, b_var = [], [], []
    for i in range(dataset.__len__()):
        img, _ = dataset.__getitem__(index=i)
        r = np.array(img[0, :, :])
        g = np.array(img[1, :, :])
        b = np.array(img[2, :, :])

        r_mean.append(np.mean(r))
        g_mean.append(np.mean(g))
        b_mean.append(np.mean(b))

        r_var.append(np.var(r))
        g_var.append(np.var(g))
        b_var.append(np.var(b))

    print(f"The MEAN are: R - {np.mean(r_mean)}, G - {np.mean(g_mean)}, B - {np.mean(b_mean)}\n"
          f"The VAR are: R - {np.mean(r_var)}, G - {np.mean(g_var)}, B - {np.mean(b_var)}")


def compute_MeanImg(dataset, save_model_root):
    r, g, b = [], [], []
    for i in range(dataset.__len__()):
        img, _ = dataset.__getitem__(index=i)
        r.append(np.array(img[0, :, :]))
        g.append(np.array(img[1, :, :]))
        b.append(np.array(img[2, :, :]))

    r_sum = np.mean(np.stack(r, axis=-1), axis=-1)
    g_sum = np.mean(np.stack(g, axis=-1), axis=-1)
    b_sum = np.mean(np.stack(b, axis=-1), axis=-1)
    mean_img = torch.moveaxis(torch.from_numpy(np.stack([r_sum, g_sum, b_sum], axis=-1)), -1, 0)
    np.save(save_model_root + 'nyu_Mimg.npy', mean_img)

    print("Process Completed")

def blockPrint():
    """
        Deviate the standard output channel to null
    """
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    """
        Restore the standard output channel
    """
    sys.stdout = old_stout

def print_device_info(os_type):
    """
        Test are on CPU, this function outputs its model name
    """
    if(os_type=='w'):
        dev = os.popen("wmic cpu get name").read()
        dev = str(dev).strip().split("\n")[-1]
    else:
        dev = os.popen("lscpu |grep 'Model name'")
        dev = str(dev).strip("]'").split(":")[1].strip()
    
    print("Experiment runned throught CPU:", dev)

def debug_print():
    """
        Sometimes blockPrint blocks too much, run this 2/3 times to reset print
    """
    enablePrint()
    print("Debug")
########################################################## Data ##########################################################

#---------------------------------------------------- Data augmentation -------------------------------------------------#

def pixel_shift(depth_img, shift):
    depth_img = depth_img + shift
    return depth_img


def random_crop(x, y, crop_size=(192, 256)):
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    h, w, _ = x.shape
    rangew = (w - crop_size[0]) // 2 if w > crop_size[0] else 0
    rangeh = (h - crop_size[1]) // 2 if h > crop_size[1] else 0
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    cropped_x = x[offseth:offseth + crop_size[0], offsetw:offsetw + crop_size[1], :]
    cropped_y = y[offseth:offseth + crop_size[0], offsetw:offsetw + crop_size[1], :]
    cropped_y = cropped_y[:, :, ~np.all(cropped_y == 0, axis=(0, 1))]
    if cropped_y.shape[-1] == 0:
        return x, y
    else:
        return cropped_x, cropped_y


def augmentation2D(img, depth, print_info_aug):
    # Random flipping
    if random.uniform(0, 1) <= augmentation_parameters['flip']:
        img = (img[..., ::1, :, :]).copy()
        depth = (depth[..., ::1, :, :]).copy()
        if print_info_aug:
            print('--> Random flipped')
    # Random mirroring
    if random.uniform(0, 1) <= augmentation_parameters['mirror']:
        img = (img[..., ::-1, :]).copy()
        depth = (depth[..., ::-1, :]).copy()
        if print_info_aug:
            print('--> Random mirrored')
    # Augment image
    if random.uniform(0, 1) <= augmentation_parameters['color&bright']:
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        img = img ** gamma
        brightness = random.uniform(0.9, 1.1)
        img = img * brightness
        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((img.shape[0], img.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        img *= color_image
        img = np.clip(img, 0, 255)  # Originally with 0 and 1
        if print_info_aug:
            print('--> Image randomly augmented')
    # Channel swap
    if random.uniform(0, 1) <= augmentation_parameters['c_swap']:
        indices = list(product([0, 1, 2], repeat=3))
        policy_idx = random.randint(0, len(indices) - 1)
        img = img[..., list(indices[policy_idx])]
        if print_info_aug:
            print('--> Channel swapped')
    # Random crop
    if random.random() <= augmentation_parameters['random_crop']:
        img, depth = random_crop(img, depth)
        if print_info_aug:
            print('--> Random cropped')
    # Depth Shift
    if random.random() <= augmentation_parameters['random_d_shift']:
        random_shift = random.randint(-10, 10)
        depth = pixel_shift(depth, shift=random_shift)
        if print_info_aug:
            print('--> Depth Shifted of {} cm'.format(random_shift))

    return img, depth

#---------------------------------------------------- Dataset -------------------------------------------------#

class NYU2_Dataset:
    """
      * Indoor img (480, 640, 3) depth (480, 640, 1) both in png -> range between 0.5 to 10 meters
      * 654 Test and 50688 Train images
    """

    def __init__(self, path, dts_type, aug, rgb_h_res, d_h_res, dts_size=0, scenarios='indoor'):
        self.dataset = path
        self.x = []
        self.y = []
        self.info = 0
        self.dts_type = dts_type
        self.aug = aug
        self.rgb_h_res = rgb_h_res
        self.d_h_res = d_h_res
        self.scenarios = scenarios

        # Handle dataset
        if self.dts_type == 'test':
            img_path = self.dataset + self.dts_type + '/eigen_test_rgb.npy'
            depth_path = self.dataset + self.dts_type + '/eigen_test_depth.npy'

            rgb = np.load(img_path)
            depth = np.load(depth_path)

            self.x = rgb
            self.y = depth

            if dts_size != 0:
                self.x = rgb[:dts_size]
                self.y = depth[:dts_size]

            self.info = len(self.x)

        elif self.dts_type == 'train':
            scenarios = os.listdir(self.dataset + self.dts_type + '/')
            for scene in scenarios:
                elem = os.listdir(self.dataset + self.dts_type + '/' + scene)
                for el in elem:
                    if 'jpg' in el:
                        self.x.append(self.dts_type + '/' + scene + '/' + el)
                    elif 'png' in el:
                        self.y.append(self.dts_type + '/' + scene + '/' + el)
                    else:
                        raise SystemError('Type image error (train)')

            if len(self.x) != len(self.y):
                raise SystemError('Problem with Img and Gt, no same train_size')

            self.x.sort()
            self.y.sort()

            if dts_size != 0:
                self.x = self.x[:dts_size]
                self.y = self.y[:dts_size]

            self.info = len(self.x)

        else:
            raise SystemError('Problem in the path')

    def __len__(self):
        return self.info

    def __getitem__(self, index=None, print_info_aug=False):
        if index is None:
            index = np.random.randint(0, self.info)

        # Load Image
        if self.dts_type == 'test':
            img = self.x[index]
        else:
            img = Image.open(self.dataset + self.x[index]).convert('RGB')
            img = np.array(img)

        # Load Depth Image
        if self.dts_type == 'test':
            depth = np.expand_dims(self.y[index] * 100, axis=-1)
        else:
            depth = Image.open(self.dataset + self.y[index])
            depth = np.array(depth) / 255
            depth = np.clip(depth * 1000, 50, 1000)
            depth = np.expand_dims(depth, axis=-1)

        # Augmentation
        if self.aug:
            img, depth = augmentation2D(img, depth, print_info_aug)

        img_post_processing = TT.Compose([
            TT.ToTensor(),
            TT.Resize((global_var['RGB_img_res'][1], global_var['RGB_img_res'][2]), antialias=True),
            TT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Imagenet
        ])
        depth_post_processing = TT.Compose([
            TT.ToTensor(),
            TT.Resize((global_var['D_img_res'][1], global_var['D_img_res'][2]), antialias=True),
        ])

        img = img_post_processing(img/255)
        depth = depth_post_processing(depth)

        return img.float(), depth.float()

#---------------------------------------------------- Dataloader -------------------------------------------------#

def init_train_test_loader(dts_type, dts_root_path, rgb_h_res, d_h_res, bs_train, bs_eval, num_workers, size_train=0, size_test=0):
    if dts_type == 'nyu':
        Dataset_class = NYU2_Dataset
        dts_root_path = dts_root_path + 'NYUv2/'
    else:
        print('OCCHIO AL DATASET')


    # Load Datasets
    test_Dataset = Dataset_class(
        path=dts_root_path, dts_type='test', aug=False, rgb_h_res=rgb_h_res, d_h_res=d_h_res, dts_size=size_test
    )
    training_Dataset = Dataset_class(
        path=dts_root_path, dts_type='train', aug=True, rgb_h_res=rgb_h_res, d_h_res=d_h_res, dts_size=size_train
    )
    # Create Dataloaders
    training_DataLoader = DataLoader(
        training_Dataset, batch_size=bs_train, shuffle=True, pin_memory=True, num_workers=num_workers
    )
    test_DataLoader = DataLoader(
        test_Dataset, batch_size=bs_eval, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return training_DataLoader, test_DataLoader, training_Dataset, test_Dataset

########################################################## Loss function ##########################################################

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs

    return ret


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out


class balanced_loss_function(nn.Module):

    def __init__(self, device):
        super(balanced_loss_function, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=0)
        self.get_gradient = Sobel().to(device)
        self.device = device

    def forward(self, output, depth):
        with torch.no_grad():
            ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().to(self.device)

        depth_grad = self.get_gradient(depth)
        output_grad = self.get_gradient(output)

        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        loss_depth = torch.abs(output - depth).mean()
        loss_dx = torch.abs(output_grad_dx - depth_grad_dx).mean()
        loss_dy = torch.abs(output_grad_dy - depth_grad_dy).mean()
        loss_normal = 100 * torch.abs(1 - self.cos(output_normal, depth_normal)).mean()

        loss_ssim = (1 - ssim(output, depth, val_range=1000.0)) * 100

        loss_grad = (loss_dx + loss_dy) / 2

        return loss_depth, loss_ssim, loss_normal, loss_grad

########################################################## Architecture ##########################################################

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()  # nn.SiLU()
    )


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, device, stride=1, depth=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels * depth,
                                   kernel_size=kernel_size,
                                   groups=depth,
                                   padding=1,
                                   stride=stride,
                                   bias=bias).to(device)
        self.pointwise = nn.Conv2d(out_channels * depth, out_channels, kernel_size=(1, 1), bias=bias).to(device)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        # nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        SeparableConv2d(in_channels=inp, out_channels=oup, kernel_size=kernal_size, stride=stride,
                        bias=False, device='cpu'),
        nn.BatchNorm2d(oup),
        nn.ReLU()  # nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),  # nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        # head_dim = dim // heads
        self.scale = dim_head ** -0.5
        # print("------------------------------------DIM--------------------------", dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.attn_drop = nn.Dropout(0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0)

        self.sr_ratio = 4
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C, W = x.shape # torch.Size([1, 4, 192, 144])

        q = self.q(x)

        if self.sr_ratio > 1:
            # x_ = x.permute(0, 2, 1).reshape(B, C, N, W)
            x_ = x.reshape(B, W, N, C)
            # print("-------------------------------------------------------------------", x_.size)
            x_ = self.sr(x_).reshape(B, -1, self.dim).permute(0, 2, 1)
            # print("-------------------------------------------------------------------", x_.size)
            x_ = self.norm(x_.permute(0, 2, 1))
            kv = self.kv(x_).reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # q = q.reshape([q.shape[0], q.shape[1], q.shape[2]*(q.shape[3]//k.shape[3]), k.shape[3]])
        q = q.reshape([q.shape[0], q.shape[1], (q.shape[0]*q.shape[1]*q.shape[2]*q.shape[3])//(q.shape[0]*q.shape[1]*k.shape[3]), k.shape[3]])    # use this for xxs architecture
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C, W)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),  # nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),  # nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),  # nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)  # Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        #print("*********************************** Start logging ***********************************")
        #print("Transformer input shape: ",x.shape)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        #print("Rearranged input shape: ",x.shape)

        start_time = perf_counter() ############################## Time measurament
        x = self.transformer(x)
        end_time = perf_counter() ############################## Time measurament

        #print("Transformer output shape: ",x.shape)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)
        #print("Rearranged output shape: ",x.shape)
        #print("**************************************************************************************")
        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x, end_time-start_time ############################## Time measurament


class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes,transformer_times, sample_cnt, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        self.transformer_times = transformer_times ############################## Time measurament
        self.sample_cnt = sample_cnt ############################## Time measurament

        L = [1, 1, 1]  # L = [2, 4, 3] # --> +5 FPS

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))  # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))

        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        # self.pool = nn.AvgPool2d(ih // 32, 1)
        # self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        y0 = self.conv1(x)
        x = self.mv2[0](y0)

        y1 = self.mv2[1](x)
        x = self.mv2[2](y1)
        x = self.mv2[3](x)  # Repeat

        y2 = self.mv2[4](x)
        x,mvit_time_1 = self.mvit[0](y2)
        self.transformer_times[0][self.sample_cnt] = mvit_time_1 ############################## Time measurament

        y3 = self.mv2[5](x)
        x,mvit_time_2 = self.mvit[1](y3)
        self.transformer_times[1][self.sample_cnt] = mvit_time_2 ############################## Time measurament

        x = self.mv2[6](x)
        x,mvit_time_3 = self.mvit[2](x)
        self.transformer_times[2][self.sample_cnt] = mvit_time_3 ############################## Time measurament
        x = self.conv2(x)

        self.sample_cnt += 1 ############################## Time measurament
        if(self.sample_cnt == 655):
          self.sample_cnt = 0

        return x, [y0, y1, y2, y3]


def mobilevit_xxs(transformer_times, sample_cnt): ############################## Time measurament
    enc_type = 'xxs'
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 160]  # 320
    return MobileViT((global_var['RGB_img_res'][1], global_var['RGB_img_res'][2]), dims, channels, num_classes=1000, expansion=2,
                      transformer_times=transformer_times, sample_cnt=sample_cnt), enc_type ############################## Time measurament


def mobilevit_xs(transformer_times, sample_cnt):
    enc_type = 'xs'
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 192] # 384
    return MobileViT((global_var['RGB_img_res'][1], global_var['RGB_img_res'][2]), dims, channels, num_classes=1000,
                      transformer_times=transformer_times, sample_cnt=sample_cnt), enc_type ############################## Time measurament


def mobilevit_s(transformer_times, sample_cnt):
    enc_type = 's'
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 320]
    return MobileViT((global_var['RGB_img_res'][1], global_var['RGB_img_res'][2]), dims, channels, num_classes=1000,
                     transformer_times=transformer_times, sample_cnt=sample_cnt), enc_type ############################## Time measurament


class UpSample_layer(nn.Module):
    def __init__(self, inp, oup, flag, sep_conv_filters, name, device):
        super(UpSample_layer, self).__init__()
        self.flag = flag
        self.name = name
        self.conv2d_transpose = nn.ConvTranspose2d(inp, oup, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                                   dilation=1, output_padding=(1, 1), bias=False)
        self.end_up_layer = nn.Sequential(
            SeparableConv2d(sep_conv_filters, oup, kernel_size=(3, 3), device=device),
            nn.ReLU()
        )


    def forward(self, x, enc_layer):
        x = self.conv2d_transpose(x)
        if x.shape[-1] != enc_layer.shape[-1]:
            enc_layer = torch.nn.functional.pad(enc_layer, pad=(1, 0), mode='constant', value=0.0)
        if x.shape[-1] != enc_layer.shape[-1]:
            enc_layer = torch.nn.functional.pad(enc_layer, pad=(0, 1), mode='constant', value=0.0)
        x = torch.cat([x, enc_layer], dim=1)
        x = self.end_up_layer(x)

        return x


class SPEED_decoder(nn.Module):
    def __init__(self, device, typ):
        super(SPEED_decoder, self).__init__()
        self.conv2d_in = nn.Conv2d(320 if typ == 's' else 192 if typ == 'xs' else 160,
                                   128 if typ == 's' else 128 if typ == 'xs' else 64,
                                   kernel_size=(1, 1), padding='same', bias=False)
        self.ups_block_1 = UpSample_layer(128 if typ == 's' else 128 if typ == 'xs' else 64,
                                          64 if typ == 's' else 64 if typ == 'xs' else 32,
                                          flag=True,
                                          sep_conv_filters=192 if typ == 's' else 144 if typ == 'xs' else 96,
                                          name='up1', device=device)
        self.ups_block_2 = UpSample_layer(64 if typ == 's' else 64 if typ == 'xs' else 32,
                                          32 if typ == 's' else 32 if typ == 'xs' else 16,
                                          flag=False,
                                          sep_conv_filters=128 if typ == 's' else 96 if typ == 'xs' else 64,
                                          name='up2', device=device)
        self.ups_block_3 = UpSample_layer(32 if typ == 's' else 32 if typ == 'xs' else 16,
                                          16 if typ == 's' else 16 if typ == 'xs' else 8,
                                          flag=False,
                                          sep_conv_filters=80 if typ == 's' else 64 if typ == 'xs' else 32,
                                          name='up3', device=device)
        self.conv2d_out = nn.Conv2d(16 if typ == 's' else 16 if typ == 'xs' else 8,
                                    1, kernel_size=(3, 3), padding='same', bias=False)

    def forward(self, x, enc_layer_list):
        x = self.conv2d_in(x)
        x = self.ups_block_1(x, enc_layer_list[3])
        x = self.ups_block_2(x, enc_layer_list[2])
        x = self.ups_block_3(x, enc_layer_list[1])
        x = self.conv2d_out(x)
        return x


class build_model(nn.Module):
    """
        MobileVit -> https://arxiv.org/pdf/2110.02178.pdf
    """
    def __init__(self, device, arch_type):
        super(build_model, self).__init__()
        self.transformer_times = np.zeros((3,655),dtype='float') ############################## Time measurament
        self.sample_cnt = 0 ############################## Time measurament

        if arch_type == 's':
            self.encoder, enc_type = mobilevit_s(self.transformer_times, self.sample_cnt) ############################## Time measurament
        elif arch_type == 'xs':
            self.encoder, enc_type = mobilevit_xs(self.transformer_times, self.sample_cnt) ############################## Time measurament
        else:
            self.encoder, enc_type = mobilevit_xxs(self.transformer_times, self.sample_cnt) ############################## Time measurament
        self.decoder = SPEED_decoder(device=device, typ=enc_type)

    def forward(self, x):
        x, enc_layer = self.encoder(x)
        x = self.decoder(x, enc_layer)
        return x


class SPEED_decoder(nn.Module):
    def __init__(self, device, typ):
        super(SPEED_decoder, self).__init__()
        self.conv2d_in = nn.Conv2d(320 if typ == 's' else 192 if typ == 'xs' else 160,
                                   128 if typ == 's' else 128 if typ == 'xs' else 64,
                                   kernel_size=(1, 1), padding='same', bias=False)
        self.ups_block_1 = UpSample_layer(128 if typ == 's' else 128 if typ == 'xs' else 64,
                                          64 if typ == 's' else 64 if typ == 'xs' else 32,
                                          flag=True,
                                          sep_conv_filters=192 if typ == 's' else 144 if typ == 'xs' else 96,
                                          name='up1', device=device)
        self.ups_block_2 = UpSample_layer(64 if typ == 's' else 64 if typ == 'xs' else 32,
                                          32 if typ == 's' else 32 if typ == 'xs' else 16,
                                          flag=False,
                                          sep_conv_filters=128 if typ == 's' else 96 if typ == 'xs' else 64,
                                          name='up2', device=device)
        self.ups_block_3 = UpSample_layer(32 if typ == 's' else 32 if typ == 'xs' else 16,
                                          16 if typ == 's' else 16 if typ == 'xs' else 8,
                                          flag=False,
                                          sep_conv_filters=80 if typ == 's' else 64 if typ == 'xs' else 32,
                                          name='up3', device=device)
        self.conv2d_out = nn.Conv2d(16 if typ == 's' else 16 if typ == 'xs' else 8,
                                    1, kernel_size=(3, 3), padding='same', bias=False)

    def forward(self, x, enc_layer_list):
        x = self.conv2d_in(x)
        x = self.ups_block_1(x, enc_layer_list[3])
        x = self.ups_block_2(x, enc_layer_list[2])
        x = self.ups_block_3(x, enc_layer_list[1])
        x = self.conv2d_out(x)
        return x


class build_model(nn.Module):
    """
        MobileVit -> https://arxiv.org/pdf/2110.02178.pdf
    """
    def __init__(self, device, arch_type):
        super(build_model, self).__init__()
        self.transformer_times = np.zeros((3,655),dtype='float') ############################## Time measurament
        self.sample_cnt = 0 ############################## Time measurament

        if arch_type == 's':
            self.encoder, enc_type = mobilevit_s(self.transformer_times, self.sample_cnt) ############################## Time measurament
        elif arch_type == 'xs':
            self.encoder, enc_type = mobilevit_xs(self.transformer_times, self.sample_cnt) ############################## Time measurament
        else:
            self.encoder, enc_type = mobilevit_xxs(self.transformer_times, self.sample_cnt) ############################## Time measurament
        self.decoder = SPEED_decoder(device=device, typ=enc_type)

    def forward(self, x):
        x, enc_layer = self.encoder(x)
        x = self.decoder(x, enc_layer)
        return x

########################################################## Evaluation metrics ##########################################################

def log10(x):
    return torch.log(x) / math.log(10)


class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3

    def evaluate(self, output, target):
        valid_mask = target > 0

        output = output[valid_mask]
        target = target[valid_mask]

        if 'kitti' in global_var['dts_type']:
            output = output[2080:] # remove first 13pixels lines
            target = target[2080:]

        abs_diff = (output - target).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0
        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0

    def update(self, result, n=1):
        self.count += n

        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_lg10 += n * result.lg10
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count,
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count)
        return avg


def compute_evaluation(test_dataloader, model, model_type, path_save_csv_results):
    best_worst_dict = {}
    result = Result()
    result.set_to_worst()
    average_meter = AverageMeter()
    model.eval()  # switch to evaluate mode

    for i, (inputs, depths) in enumerate(test_dataloader):
        #inputs, depths = inputs.cuda(), depths.cuda()
        inputs, depths = inputs.cpu(), depths.cpu()
        # compute output
        with torch.no_grad():
            predictions = model(inputs)
        result.evaluate(predictions, depths)
        average_meter.update(result)  # (result, inputs.size(0))
        best_worst_dict[i] = result.rmse

    avg = average_meter.average()

    print('MAE={average.mae:.3f}\n'
          'RMSE={average.rmse:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'REL={average.absrel:.3f}\n'
          'Lg10={average.lg10:.3f}'.format(average=avg))

    with open(path_save_csv_results + 'test' + model_type + 'results.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['mse', 'rmse', 'absrel', 'lg10', 'mae', 'delta1', 'delta2', 'delta3'])
        writer.writeheader()
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                         'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3})

    return best_worst_dict, avg

########################################################## Train ##########################################################

def process(device):
    # Set-seed
    os.environ["PYTHONHASHSEED"] = str(global_var['seed'])
    np.random.seed(global_var['seed'])
    torch.cuda.manual_seed(global_var['seed'])
    # Datasets loading
    training_DataLoader, test_DataLoader, training_Dataset, test_Dataset = init_train_test_loader(
        dts_type=global_var['dts_type'],
        dts_root_path=dataset_root,
        rgb_h_res=global_var['RGB_img_res'][1],
        d_h_res=global_var['D_img_res'][1],
        bs_train=global_var['batch_size'],
        bs_eval=global_var['batch_size_eval'],
        num_workers=global_var['n_workers'],
        size_train=global_var['size_train'],
        size_test=global_var['size_test']
    )
    print('INFO: There are {} training and {} testing samples'.format(training_Dataset.__len__(), test_Dataset.__len__()))
    # Prints samples
    if global_var['do_prints']:
        print(' --- Test samples --- ')
        print_img(test_Dataset, label='rgb_sample', quantity=2,
                  save_model_root=save_model_root)
        print(' --- Training augmented samples --- ')
        print_img(training_Dataset, label='aug_sample', quantity=5, print_info_aug=True,
                  save_model_root=save_model_root)
    if global_var['do_train']:
        torch.cuda.empty_cache()
        # Globals
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lrs': [], 'test_rmse': [],
                   'l_mae': [], 'l_norm': [], 'l_grad': [], 'l_ssim': []}
        min_rmse = float('inf')
        min_acc = 0
        train_loss_list = []
        test_loss_list = []
        # Loss
        criterion = balanced_loss_function(device=device)
        # Model
        model = build_model(device=device, arch_type=global_var['architecture_type']).to(device=device)
        if global_var['do_pretrained'] or global_var['imagenet_w_init']:
            model, _ = load_pretrained_model(model=model,
                                             path_weigths=save_model_root + 'build_model_best' if global_var['do_pretrained']
                                                          else imagenet_init + global_var['imagenet_enc'] + '/build_model_best',
                                             device=device,
                                             do_pretrained=global_var['do_pretrained'],
                                             imagenet_w_init=global_var['imagenet_w_init'])
        model_name = model.__class__.__name__
        if global_var['do_print_model']:
            print_model(model=model, device=device, save_model_root=save_model_root, input_shape=global_var['RGB_img_res'])
        print('The {} model has: {} trainable parameters'.format(model_name, count_parameters(model)))
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=global_var['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False
        )
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=global_var['lr_patience'], threshold=1e-4, threshold_mode='rel',
            cooldown=0, min_lr=1e-8, eps=1e-08, verbose=False
        )
        # Early stopping
        trigger_times, early_stopping_epochs = 0, global_var['e_stop_epochs']
        print("Start training: {}\n".format(model_name))
        # Train
        for epoch in range(global_var['epochs']):
            iter = 1
            model.train()
            running_loss, accuracy = 0, 0
            running_l_mae, running_l_grad, running_l_norm, running_l_ssim = 0, 0, 0, 0
            with tqdm(training_DataLoader, unit="step", position=0, leave=True) as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch + 1}/{global_var['epochs']} - Training")
                    # Load data
                    inputs, depths = batch[0].to(device=device), batch[1].to(device=device)
                    # Forward
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    # Compute loss
                    loss_depth, loss_ssim, loss_normal, loss_grad = criterion(outputs, depths)
                    loss = loss_depth + loss_normal + loss_grad + loss_ssim
                    # Backward
                    loss.backward()
                    optimizer.step()
                    # Evaluation and Stats
                    running_loss += loss.item()
                    running_l_mae += loss_depth.item()
                    running_l_norm += loss_normal.item()
                    running_l_grad += loss_grad.item()
                    running_l_ssim += loss_ssim.item()

                    train_loss_support = [loss_depth.item(), loss_normal.item(), loss_grad.item(), loss.item()]
                    train_loss_list.append(train_loss_support)

                    accuracy += compute_accuracy(outputs, depths)
                    tepoch.set_postfix({'Loss': running_loss / iter,
                                        'Acc': accuracy.item() / iter,
                                        'Lr': global_var['lr'] if not history['lrs'] else history['lrs'][-1],
                                        'L_mae': running_l_mae / iter,
                                        'L_norm': running_l_norm / iter,
                                        'L_grad': running_l_grad / iter,
                                        'L_ssim': running_l_ssim / iter
                                        })
                    iter += 1

            # Validation
            iter = 1
            model.eval()
            test_loss, test_accuracy, test_rmse = 0, 0, 0
            with tqdm(test_DataLoader, unit="step", position=0, leave=True) as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch + 1}/{global_var['epochs']} - Validation")
                    inputs, depths = batch[0].to(device=device), batch[1].to(device=device)
                    # Validation loop
                    with torch.no_grad():
                        outputs = model(inputs)
                        # Evaluation metrics
                        test_accuracy += compute_accuracy(outputs, depths)
                        # Loss
                        loss_depth, loss_ssim, loss_normal, loss_grad = criterion(outputs, depths)
                        loss = loss_depth + loss_normal + loss_grad + loss_ssim
                        test_loss += loss.item()

                        test_loss_support = [loss_depth.item(), loss_normal.item(), loss_grad.item(), loss.item()]
                        test_loss_list.append(test_loss_support)

                        # RMSE
                        test_rmse += compute_rmse(outputs, depths)
                        tepoch.set_postfix({'Loss': test_loss / iter, 'Acc': test_accuracy.item() / iter,
                                            'RMSE': test_rmse.item() / iter})
                        iter += 1

            # Update history infos
            history['lrs'].append(get_lr(optimizer))
            history['train_loss'].append(running_loss / len(training_DataLoader))
            history['val_loss'].append(test_loss / len(test_DataLoader))
            history['train_acc'].append(accuracy.item() / len(training_DataLoader))
            history['val_acc'].append(test_accuracy.item() / len(test_DataLoader))
            history['test_rmse'].append(test_rmse.item() / len(test_DataLoader))
            # Update history losses infos
            history['l_mae'].append(running_l_mae / len(training_DataLoader))
            history['l_norm'].append(running_l_norm / len(training_DataLoader))
            history['l_grad'].append(running_l_grad / len(training_DataLoader))
            history['l_ssim'].append(running_l_ssim / len(training_DataLoader))
            # Update scheduler LR
            scheduler.step(history['test_rmse'][-1])
            # Save model by best RMSE
            if min_rmse >= (test_rmse / len(test_DataLoader)):
                trigger_times = 0
                min_rmse = test_rmse / len(test_DataLoader)
                save_checkpoint(model, model_name + '_best', save_model_root)
                print('New best RMSE: {:.3f} at epoch {}'.format(min_rmse, epoch + 1))
            else:
                trigger_times += 1
                print('RMSE did not improved, EarlyStopping from {} epochs'.format(early_stopping_epochs - trigger_times))
            # Save model by best ACCURACY
            if min_acc <= (test_accuracy / len(test_DataLoader)):
                min_acc = test_accuracy / len(test_DataLoader)
                save_checkpoint(model, model_name + '_best_acc', save_model_root)
                print('New best ACCURACY: {:.3f} at epoch {}'.format(min_acc, epoch + 1))
                if trigger_times > 4:
                    trigger_times = trigger_times - 2
                    print(f"EarlyStopping increased due to Accuracy, stop in {early_stopping_epochs - trigger_times} epochs")

            save_prediction_examples(model, dataset=test_Dataset, device=device, indices=[0, 216, 432, 639], ep=epoch,
                                     save_path=save_model_root + 'evolution_img/')
            save_history(history, save_model_root + model_name + '_history')
            # Empty CUDA cache
            torch.cuda.empty_cache()

            if trigger_times == early_stopping_epochs:
                print('Val Loss did not imporved for {} epochs, training stopped'.format(early_stopping_epochs + 1))
                break

            # Save loss for graphs
            np.save(save_model_root + 'train.npy', np.array(train_loss_list))
            np.save(save_model_root + 'test.npy', np.array(test_loss_list))

        print('Finished Training')
        save_csv_history(model_name=model_name, path=save_model_root)
        plot_history(history, path=save_model_root)
        plot_loss_parts(history, path=save_model_root, title='Loss Components')

        if global_var['do_prints']:
            if os.path.exists(save_model_root + 'example&augment_img/'):
                shutil.rmtree(save_model_root + 'example&augment_img/')

    else:
        model = build_model(device=device, arch_type=global_var['architecture_type']).to(device=device)
        model, model_name = load_pretrained_model(model=model,
                                                  path_weigths=save_model_root + 's_build_model_best',
                                                  device=device,
                                                  do_pretrained=global_var['do_pretrained'],
                                                  imagenet_w_init=global_var['imagenet_w_init'])
        if global_var['do_print_model']:
            print_model(model=model, device=device, save_model_root=save_model_root,
                        input_shape=global_var['RGB_img_res'])
        print('The {} model has: {} trainable parameters'.format(model_name, count_parameters(model)))

    # Evaluate
    print(' --- Begin evaluation --- ')
    best_worst, avg = compute_evaluation(test_dataloader=test_DataLoader, model=model, model_type='_', path_save_csv_results=save_model_root)
    print(' --- End evaluation --- ')

    if global_var['do_print_best_worst']:
        sorted_best_worst = sorted(best_worst.items(), key=lambda item: item[1])
        save_best_worst(sorted_best_worst[0:10], type='best', model=model, dataset=test_Dataset, device=device, save_model_root=save_model_root)
        save_best_worst(sorted_best_worst[-10:], type='worst', model=model, dataset=test_Dataset, device=device, save_model_root=save_model_root)

    return

########################################################## Time tests ##########################################################

#---------------------------------------------------- Inference time -------------------------------------------------#
def inference_time(n,archs,os_type):
    meter_types = archs
    test_rounds = n
    blockPrint()
    device = hardware_check()
    enablePrint()

    print("---------------- Inference time test")
    print_device_info(os_type)

    # Set-seed
    os.environ["PYTHONHASHSEED"] = str(global_var['seed'])
    np.random.seed(global_var['seed'])
    torch.cuda.manual_seed(global_var['seed'])

    # Datasets loading
    _, test_DataLoader, _, _ = init_train_test_loader(
        dts_type=global_var['dts_type'],
        dts_root_path=dataset_root,
        rgb_h_res=global_var['RGB_img_res'][1],
        d_h_res=global_var['D_img_res'][1],
        bs_train=global_var['batch_size'],
        bs_eval=global_var['batch_size_eval'],
        num_workers=global_var['n_workers'],
        size_train=global_var['size_train'],
        size_test=global_var['size_test']
        )

    for arch_type in meter_types:
        print("##################################################### TEST - %s architecture #####################################################" % arch_type.upper())
        times = np.ndarray(shape=test_rounds,dtype='float')
        test_rmse = 0

        blockPrint()
        model = build_model(device=device, arch_type=arch_type).to(device=device)
        model, _ = load_pretrained_model(model=model,
                                         path_weigths=save_model_root + arch_type + '_build_model_best',
                                         device=device,
                                         do_pretrained=global_var['do_pretrained'],
                                         imagenet_w_init=global_var['imagenet_w_init'])
        enablePrint()


        print("Model: %s" % network_type)
        blockPrint()
        infos = summary(model,torch.ones(1,3,192,256).to(device))
        warnings.simplefilter(action='ignore', category=FutureWarning)
        enablePrint()
        print("Trainable parameters: %d" % count_parameters(model))
        print("Mult-Adds: %d" % int(infos["Mult-Adds"].sum()))

        for tests in range(test_rounds):
            # Evaluate
            blockPrint()
            start_time = perf_counter()
            _, avg = compute_evaluation(test_dataloader=test_DataLoader, model=model, model_type='_', path_save_csv_results=save_model_root)
            #torch.cuda.synchronize() # <--------------------------------- REMEMBER WITH GPU
            end_time = perf_counter()
            enablePrint()

            times[tests] = end_time - start_time

            if tests==0:
                test_rmse = avg.rmse

        print("Average test time: ",np.mean(times))
        print("Test times: ",times)
        print("Test rmse: ", test_rmse)

#---------------------------------------------------- Transformer time -------------------------------------------------#

def transformer_time(n,archs,os_type):
    meter_types = archs
    test_rounds = n
    blockPrint()
    device = hardware_check()
    enablePrint()

    print("\n\n---------------- Transformers time test")
    print_device_info(os_type)

    # Set-seed
    os.environ["PYTHONHASHSEED"] = str(global_var['seed'])
    np.random.seed(global_var['seed'])
    torch.cuda.manual_seed(global_var['seed'])

    # Datasets loading
    _, test_DataLoader, _, _ = init_train_test_loader(
        dts_type=global_var['dts_type'],
        dts_root_path=dataset_root,
        rgb_h_res=global_var['RGB_img_res'][1],
        d_h_res=global_var['D_img_res'][1],
        bs_train=global_var['batch_size'],
        bs_eval=global_var['batch_size_eval'],
        num_workers=global_var['n_workers'],
        size_train=global_var['size_train'],
        size_test=global_var['size_test']
        )

    for arch_type in meter_types:
        print("##################################################### TEST - %s architecture #####################################################" % arch_type.upper())
        times = np.ndarray(shape=(test_rounds,3,655),dtype='float')

        blockPrint()
        model = build_model(device=device, arch_type=arch_type).to(device=device)
        model, _ = load_pretrained_model(model=model,
                                                    path_weigths=save_model_root + arch_type + '_build_model_best',
                                                    device=device,
                                                    do_pretrained=global_var['do_pretrained'],
                                                    imagenet_w_init=global_var['imagenet_w_init'])
        enablePrint()


        print("Model: %s" % network_type)
        blockPrint()
        infos = summary(model,torch.ones(1,3,192,256).to(device))
        enablePrint()
        warnings.simplefilter(action='ignore', category=FutureWarning)
        print("Trainable parameters: %d" % count_parameters(model))
        print("Mult-Adds: %d" % int(infos["Mult-Adds"].sum()))

        for tests in range(test_rounds):
            # Evaluate
            blockPrint()
            _, _ = compute_evaluation(test_dataloader=test_DataLoader, model=model, model_type='_', path_save_csv_results=save_model_root)
            enablePrint()
            times[tests][0] = model.transformer_times[0]
            times[tests][1] = model.transformer_times[1]
            times[tests][2] = model.transformer_times[2]

        print("Mvit block 1 mean time: ",np.mean(times[tests][0]))
        print("Mvit block 2 mean time: ",np.mean(times[tests][1]))
        print("Mvit block 3 mean time: ",np.mean(times[tests][2]))
    
    print("\n\n")

#---------------------------------------------------- Full statistics -------------------------------------------------#
def full_statistics(archs,os_type):
    meter_types = archs
    blockPrint()
    device = hardware_check()
    enablePrint()

    print("---------------- Architecture statistics")
    print_device_info(os_type)

    # Set-seed
    os.environ["PYTHONHASHSEED"] = str(global_var['seed'])
    np.random.seed(global_var['seed'])
    torch.cuda.manual_seed(global_var['seed'])

    # Datasets loading
    _, test_DataLoader, _, _ = init_train_test_loader(
        dts_type=global_var['dts_type'],
        dts_root_path=dataset_root,
        rgb_h_res=global_var['RGB_img_res'][1],
        d_h_res=global_var['D_img_res'][1],
        bs_train=global_var['batch_size'],
        bs_eval=global_var['batch_size_eval'],
        num_workers=global_var['n_workers'],
        size_train=global_var['size_train'],
        size_test=global_var['size_test']
        )

    for arch_type in meter_types:
        print("################################################################ TEST - %s architecture ################################################################" % arch_type.upper())
        blockPrint()
        model = build_model(device=device, arch_type=arch_type).to(device=device)
        model, model_name = load_pretrained_model(model=model,
                                                    path_weigths=save_model_root + arch_type + '_build_model_best',
                                                    device=device,
                                                    do_pretrained=global_var['do_pretrained'],
                                                    imagenet_w_init=global_var['imagenet_w_init'])
        enablePrint()


        print("Model: %s" % network_type)
        blockPrint()
        infos = summary(model,torch.ones(1,3,192,256).to(device))
        enablePrint()
        warnings.simplefilter(action='ignore', category=FutureWarning)
        print("Trainable parameters: %d" % count_parameters(model))
        print("Mult-Adds: %d" % int(infos["Mult-Adds"].sum()))

        
        # Evaluate
        blockPrint()
        _, outputs = compute_evaluation(test_dataloader=test_DataLoader, model=model, model_type='_', path_save_csv_results=save_model_root)
        enablePrint()
        
        print("MSE: ", outputs.mse)
        print("RMSE: ", outputs.rmse)
        print("Absrel: ", outputs.absrel)
        print("Lg10: ", outputs.lg10)
        print("MAE: ", outputs.mae)
        print("Delta1: ", outputs.delta1)
        print("Delta2: ", outputs.delta2)
        print("Delta3: ", outputs.delta3)

    print("\n\n")

########################################################## Main ##########################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--os_type', 
                        type=str, 
                        required=True, 
                        help="Character for identify os: w = Windows, l = Linux")
    
    parser.add_argument('--run', 
                        type=str, 
                        required=True,
                        help="Type of test: trial = 1 run on xxs, real = 30 runs on s,xs,xxs")
    
    args = parser.parse_args()

    if(args.run == "real"):
        architectures = ['s','xs','xxs']
        iters = 30
    else:
        architectures = ['xxs']
        iters = 1

    inference_time(iters,architectures,args.os_type)
    transformer_time(iters,architectures,args.os_type)
    full_statistics(architectures,args.os_type)