import numpy as np
from .base_dataset import BaseDataset
from .imagefolder import MyImageFolder
import copy
import torch
from architectures.clip import get_clip_image_features
import os
import shutil
from embedding.preprocess import image_preprocess


class TwoClassImageFolderSubset(BaseDataset):
    def __init__(self, father_dataset: MyImageFolder,
                 class_idx: int, sub_class_idx: int = None):

        self.father_dataset = father_dataset
        self.raw_init = None
        self.class_idx = class_idx
        self.sub_class_idx = sub_class_idx
        super(TwoClassImageFolderSubset, self).__init__(self.father_dataset.DATA_PATH, self.father_dataset.SUBSET)

    def load_data(self):
        # main category
        self.DATA_INFOS['train_full_main_category'] = \
            [{'no': elem_dict['no'], 'img': elem_dict['img'], 'gt_label': 0, 'aesthetic_score': 5.0}
             for elem_dict in self.father_dataset.DATA_INFOS['train_full']
             if int(elem_dict['gt_label']) == self.class_idx]
        self.DATA_INFOS['train_init_main_category'] = \
            [{'no': elem_dict['no'], 'img': elem_dict['img'], 'gt_label': 0,  'aesthetic_score': 10.0}
             for elem_dict in self.father_dataset.DATA_INFOS['train_init']
             if int(elem_dict['gt_label']) == self.class_idx]
        # sub category and other categories (if exists)
        if self.sub_class_idx is not None:
            self.DATA_INFOS['train_sub_category'] = \
                [{'no': elem_dict['no'], 'img': elem_dict['img'], 'gt_label': 1, 'aesthetic_score': 5.0}
                 for elem_dict in self.father_dataset.DATA_INFOS['train_full']
                 if int(elem_dict['gt_label']) == self.sub_class_idx] + \
                [{'no': elem_dict['no'], 'img': elem_dict['img'], 'gt_label': 1, 'aesthetic_score': 5.0}
                 for elem_dict in self.father_dataset.DATA_INFOS['train_init']
                 if int(elem_dict['gt_label']) == self.sub_class_idx]
            self.DATA_INFOS['train_other_category'] = \
                [{'no': elem_dict['no'], 'img': elem_dict['img'], 'gt_label': 1, 'aesthetic_score': 5.0}
                 for elem_dict in self.father_dataset.DATA_INFOS['train_full']
                 if int(elem_dict['gt_label']) not in [self.class_idx, self.sub_class_idx]] + \
                [{'no': elem_dict['no'], 'img': elem_dict['img'], 'gt_label': 1, 'aesthetic_score': 5.0}
                 for elem_dict in self.father_dataset.DATA_INFOS['train_init']
                 if int(elem_dict['gt_label']) not in [self.class_idx, self.sub_class_idx]]
        else:
            self.DATA_INFOS['train_other_category'] = \
                [{'no': elem_dict['no'], 'img': elem_dict['img'], 'gt_label': 1, 'aesthetic_score': 5.0}
                 for elem_dict in self.father_dataset.DATA_INFOS['train_full']
                 if int(elem_dict['gt_label']) not in [self.class_idx, self.sub_class_idx]] + \
                [{'no': elem_dict['no'], 'img': elem_dict['img'], 'gt_label': 1, 'aesthetic_score': 5.0}
                 for elem_dict in self.father_dataset.DATA_INFOS['train_init']
                 if int(elem_dict['gt_label']) != self.class_idx]

        self.DATA_INFOS['train_generated_category'] = \
            [{'no': elem_dict['no'], 'img': elem_dict['img'], 'gt_label': 1, 'aesthetic_score': 5.0}
             for elem_dict in self.father_dataset.DATA_INFOS['train_generated']
             if int(elem_dict['gt_label']) == self.class_idx]

        self.INDEX_LB = np.zeros(len(self.DATA_INFOS['train_full_main_category']), dtype=bool)
        self.CLASSES = [self.father_dataset.CLASSES[self.class_idx], 'others']

    def prepare_data(self, idx, split, transform=None, aug_transform=None, task='scoring'):
        x_path, y = self.DATA_INFOS[split][idx]['img'], self.DATA_INFOS[split][idx]['gt_label']

        x = self.father_dataset.raw_full.loader(x_path)
        if aug_transform is not None:
            x = aug_transform(x)

        # extract clip embedding
        with torch.no_grad():
            if task == 'scoring':
                x = get_clip_image_features(x).to(torch.float)
            else:
                x = get_clip_image_features(x, task='cls').to(torch.float)

        return x, y, self.DATA_INFOS[split][idx]['no'], idx, \
               self.DATA_INFOS[split][idx]['aesthetic_score'] if 'aesthetic_score' in self.DATA_INFOS[split][
                   idx].keys() else 0.0

    # Add new labeled samples
    def update_lb(self, new_lb):
        self.INDEX_LB[new_lb] = True
        self.DATA_INFOS['train'] = copy.deepcopy(
            list(np.array(self.DATA_INFOS['train_full_main_category'])[self.INDEX_LB])) + \
                                   copy.deepcopy(self.DATA_INFOS['train_init_main_category'])
        self.DATA_INFOS['train_full'] = copy.deepcopy(self.DATA_INFOS['train_full_main_category']) + \
                                        copy.deepcopy(self.DATA_INFOS['train_init_main_category']) + \
                                        copy.deepcopy(self.DATA_INFOS['train_other_category'])
        train_length = len(self.DATA_INFOS['train'])
        if 'train_sub_category' in self.DATA_INFOS.keys():
            self.DATA_INFOS['train'] += list(np.random.choice(self.DATA_INFOS['train_sub_category'],
                                                              min(train_length,
                                                                  len(self.DATA_INFOS['train_sub_category'])),
                                                              replace=False))
            self.DATA_INFOS['train_full'] += self.DATA_INFOS['train_sub_category']
        self.DATA_INFOS['train'] += list(np.random.choice(self.DATA_INFOS['train_other_category'],
                                                          min(train_length,
                                                              len(self.DATA_INFOS['train_other_category'])),
                                                          replace=False))
        self.DATA_INFOS['train'] += list(np.random.choice(self.DATA_INFOS['train_generated_category'],
                                                          min(train_length,
                                                              len(self.DATA_INFOS['train_generated_category'])),
                                                          replace=False))
        self.select_ulb()
        self.QUERIED_HISTORY.append(list(new_lb))

    def initialize_lb(self):
        self.DATA_INFOS['train'] = copy.deepcopy(self.DATA_INFOS['train_init_main_category'])
        train_length = len(self.DATA_INFOS['train'])
        if 'train_sub_category' in self.DATA_INFOS.keys():
            self.DATA_INFOS['train'] += list(np.random.choice(self.DATA_INFOS['train_sub_category'],
                                                              min(train_length,
                                                                  len(self.DATA_INFOS['train_sub_category'])),
                                                              replace=False))
            self.DATA_INFOS['train_full'] += self.DATA_INFOS['train_sub_category']
        self.DATA_INFOS['train'] += list(np.random.choice(self.DATA_INFOS['train_other_category'],
                                                          min(train_length,
                                                              len(self.DATA_INFOS['train_other_category'])),
                                                          replace=False))
        self.DATA_INFOS['train'] += list(np.random.choice(self.DATA_INFOS['train_generated_category'],
                                                          min(train_length,
                                                              len(self.DATA_INFOS['train_generated_category'])),
                                                          replace=False))
        self.select_ulb()

    def select_ulb(self):
        U_TEMP = np.arange(len(self.DATA_INFOS['train_full_main_category']))[~self.INDEX_LB]
        if self.SUBSET >= len(self.DATA_INFOS['train_full_main_category']) - len(self.DATA_INFOS['train']):
            U_SELECTED = U_TEMP
        else:
            U_SELECTED = np.random.choice(U_TEMP, self.SUBSET, replace=False)
        self.INDEX_ULB = np.array([True if x in U_SELECTED else False
                                   for x in range(len(self.DATA_INFOS['train_full_main_category']))])
        self.DATA_INFOS['train_u'] = np.array(self.DATA_INFOS['train_full_main_category'])[self.INDEX_ULB]

    def move_selected_images(self, url, temp_path=None):
        if temp_path is None:
            temp_original_path = os.path.join(self.DATA_PATH, '~Temp_selected_images', 'original')
            temp_processed_path = os.path.join(self.DATA_PATH, '~Temp_selected_images', 'processed')
        else:
            temp_original_path = os.path.join(temp_path, 'original')
            temp_processed_path = os.path.join(temp_path, 'processed')
        if os.path.exists(temp_original_path):
            shutil.rmtree(temp_original_path)
        if os.path.exists(temp_processed_path):
            shutil.rmtree(temp_processed_path)
        print("Preprocess datasets")
        os.makedirs(temp_original_path, mode=0o777, exist_ok=True)
        selected_images = self.DATA_INFOS['train_init_main_category'] + \
                          list(np.array(self.DATA_INFOS['train_full_main_category'])[self.INDEX_LB])
        for data_dict in selected_images:
            shutil.copy(data_dict['img'], temp_original_path)
        os.makedirs(temp_processed_path, mode=0o777, exist_ok=True)
        image_preprocess(url, self.CLASSES[0], temp_original_path, temp_processed_path,
                         768, 768, "ignore", False, False, False)
        print("Preprocess done!")
        return temp_processed_path
