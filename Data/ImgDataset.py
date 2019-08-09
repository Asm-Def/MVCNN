import numpy as np
import glob
import torch.utils.data
import os
import math
from PIL import Image
import torch
import torchvision as vision
from torchvision import transforms, datasets
import random
from Data.classnames import *

def get_ind(filename: str):
    filename = os.path.split(filename)[-1]
    ind = filename.split('.')[0].rsplit('_', 2)[1:]
    return int(ind[0]) * 12 + int(ind[1])

class MultiviewImgDataset(torch.utils.data.Dataset):
    # TODO
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, num_models=0, num_views=12, shuffle=True):
        self.classnames = classnames
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views

        parent_dir, set_ = os.path.split(root_dir)
        print('set_=', set_)
        print('parent_dir=', parent_dir)
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(os.path.join(parent_dir, set_, self.classnames[i], self.classnames[i]+'_*_*.png')), key=get_ind)
            stride = int(12/self.num_views) # 12 6 4 3 2 1
            all_files = all_files[::stride]

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            self.filepaths = filepaths_new


        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return int(len(self.filepaths)/self.num_views)


    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])



class SingleImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, num_models=0, num_views=12):
        self.classnames = classnames
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode

        parent_dir, set_ = os.path.split(root_dir)
        print('set_=', set_)
        print('parent_dir=', parent_dir)
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(os.path.join(parent_dir, set_, self.classnames[i], self.classnames[i]+'_*_*.png')), key=get_ind)
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = os.path.split(os.path.split(path)[0])[1]
        class_id = self.classnames.index(class_name)

        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)

        return (class_id, im, path)


