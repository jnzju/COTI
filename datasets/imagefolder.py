from torchvision import datasets
import torch
import numpy as np
from .builder import DATASETS
from .base_dataset import BaseDataset
import os.path as osp
import os
from .generated_dataset import generate_virtual_dataset
from architectures.clip import get_clip_image_features


@DATASETS.register_module()
class MyImageFolder(BaseDataset):
    def __init__(self,
                 data_path=None,
                 subset=None,
                 initial_generated_images_per_class=1000,
                 url="http://127.0.0.1:7860"):
        self.raw_full = None
        self.raw_init = None
        self.SUB_CATEGORY = None
        self.url = url
        self.initial_generated_images_per_class = initial_generated_images_per_class
        super(MyImageFolder, self).__init__(data_path, subset)

    def load_data(self):
        if self.DATA_PATH is None:
            self.DATA_PATH = osp.join(os.path.abspath('..'), 'data', 'stable_diffusion_dataset')
        data_path_full = osp.join(self.DATA_PATH, 'training_dataset')
        data_path_initial = osp.join(self.DATA_PATH, 'training_dataset_initial')
        self.raw_full = datasets.ImageFolder(data_path_full)
        self.raw_init = datasets.ImageFolder(osp.join(data_path_initial))
        num_full = len(self.raw_full.targets)
        num_init = len(self.raw_init.targets)
        self.DATA_INFOS['train_full'] = [{'no': i, 'img': self.raw_full.imgs[i][0],
                                          'gt_label': self.raw_full.targets[i]} for i in range(num_full)]
        self.DATA_INFOS['train_init'] = [{'no': i, 'img': self.raw_init.imgs[i][0],
                                          'gt_label': self.raw_init.targets[i]} for i in range(num_init)]

        self.INDEX_LB = np.zeros(num_full, dtype=bool)
        self.CLASSES = self.raw_full.classes
        # mapping
        # axolotl <-> frilled_lizard
        # crampfish <-> garfish
        self.SUB_CATEGORY = {0: 3, 1: 4, 2: 6, 3: 0, 4: 1, 5: 8, 6: 2, 7: 9, 8: 5, 9: 7}

        # 如果没有生成过虚拟样本，现场生成
        generated_dataset_path = os.path.join(self.DATA_PATH, 'training_dataset_generated')
        if not os.path.exists(generated_dataset_path):
            os.makedirs(generated_dataset_path, mode=0o777, exist_ok=True)
        for category in self.CLASSES:
            generated_dataset_path_category = \
                os.path.join(generated_dataset_path, category)
            if not os.path.exists(generated_dataset_path_category):
                os.makedirs(generated_dataset_path_category, mode=0o777, exist_ok=True)
                self.DATA_INFOS['train_generated_category'] = generate_virtual_dataset(
                    url=self.url,
                    prompt=category + f", a_photo_of_{category}" + ", real_life",
                    num_samples=self.initial_generated_images_per_class, temp_dir=generated_dataset_path_category)
        # 读取生成样本
        self.raw_generated = datasets.ImageFolder(osp.join(generated_dataset_path))
        self.DATA_INFOS['train_generated'] = [{'no': i, 'img': self.raw_generated.imgs[i][0],
                                               'gt_label': self.raw_generated.targets[i]} for i in range(num_init)]

    # Only used for loading data
    def prepare_data(self, idx, split, transform=None, aug_transform=None):
        x_path, y = self.DATA_INFOS[split][idx]['img'], self.DATA_INFOS[split][idx]['gt_label']

        x = self.raw_full.loader(x_path)
        if aug_transform is not None:
            x = aug_transform(x)

        # extract clip embedding
        with torch.no_grad():
            x = get_clip_image_features(x).to(torch.float)

        return x, y, self.DATA_INFOS[split][idx]['no'], idx, \
               self.DATA_INFOS[split][idx]['aesthetic_score'] if 'aesthetic_score' in self.DATA_INFOS[split][idx].keys() else 0.0
