from abc import ABCMeta
import random
import numpy as np
import copy
import torch
from architectures.clip import get_clip_image_features
import os
import os.path as osp
from architectures.clip import get_clip_image_features
from torchvision import datasets
from PIL import Image

def subset(alist, idxs) -> list:
        sub_list = []
        for idx in idxs:
            sub_list.append(alist[idx])
            
        return sub_list
    
def split_list(alist, group_num, shuffle) -> list:
    index = list(range(len(alist)))
    
    if shuffle:
        random.shuffle(index)
    
    elem_num = len(alist) // group_num # 每一个子列表所含有的元素数量
    sub_lists = []
    
    # 取出每一个子列表所包含的元素，存入字典中
    for idx in range(group_num):
        start, end = idx*elem_num, (idx+1)*elem_num
        sub_lists.append(subset(alist, index[start:end]))
    
    # 将剩余元素合并到最后一组
    if group_num * elem_num != len(index): 
        sub_lists[-1] += subset(alist, index[end:])
    
    return sub_lists

class BaseDataset(object):

    def __init__(self, args):
        self.args = args
        self.INDEX_LB = np.zeros([], dtype=bool)
        self.QUERIED_HISTORY = []
        self.DATA_INFOS = {'train': [], 'train_full': [], 'tasks':[], 'val': []}
        self.task_idx = 0
        
        self.load_data()

    # Only used for loading data
    def prepare_data(self, idx, split, transform=None, aug_transform=None, task='scoring'):
        x_path, y = self.DATA_INFOS[split][idx]['img'], self.DATA_INFOS[split][idx]['gt_label']
        
        x = Image.open(x_path)
        
        if aug_transform is not None:
            x = aug_transform(x)

        # extract clip embedding
        with torch.no_grad():
            if task == 'scoring':
                x = get_clip_image_features(x).to(torch.float)
            else:
                x = get_clip_image_features(x, task='cls').to(torch.float)

        return x, y, self.DATA_INFOS[split][idx]['type'], idx, self.DATA_INFOS[split][idx]['aesthetic_score']

    # Used for initialize data info
    def load_data(self):
        self.raw_full = datasets.ImageFolder(self.args.training_dataset)
        self.raw_init = datasets.ImageFolder(self.args.training_dataset_initial)
        num_full = len(self.raw_full.targets)
        num_init = len(self.raw_init.targets)
        
        self.DATA_INFOS['train_init'] = [
            {'img': self.raw_init.imgs[i][0], 'gt_label': 0, 'aesthetic_score': 10.0, 'type': 'init'}
            for i in range(num_init) if self.raw_init.targets[i] == self.args.class_idx
        ]
        
        self.DATA_INFOS['train_full'] = [
            {'img': self.raw_full.imgs[i][0], 'gt_label': 0, 'aesthetic_score': 5.0 , 'type': 'full'}
            for i in range(num_full) if self.raw_full.targets[i] == self.args.class_idx
        ]
        
        #used for classifying
        self.DATA_INFOS['train_sub'] =  [
            {'img': self.raw_full.imgs[i][0], 'gt_label': 1, 'aesthetic_score': 0.0 , 'type': 'full'}
            for i in range(num_full) if self.raw_full.targets[i] == self.args.subclass_idx
        ] + [
            {'img': self.raw_init.imgs[i][0], 'gt_label': 1, 'aesthetic_score': 0.0 , 'type': 'init'}
            for i in range(num_init) if self.raw_init.targets[i] == self.args.subclass_idx
        ]
        
    def split_tasks(self):
        
        self.DATA_INFOS['tasks'] = split_list(alist=self.DATA_INFOS['train_full'], group_num=self.args.task_num, shuffle=True)
        
        self.DATA_INFOS['tasks'][0] += self.DATA_INFOS['train_init']
        
        self.task_idx = -1