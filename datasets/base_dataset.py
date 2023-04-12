from abc import ABCMeta
import numpy as np
import copy


class BaseDataset(object, metaclass=ABCMeta):
    """Base dataset.
    """

    def __init__(self,
                 data_path=None,
                 # initial_size=None,
                 subset=None):
        super(BaseDataset, self).__init__()

        self.CLASSES = None
        self.DATA_PATH = None
        self.INDEX_LB = np.zeros([], dtype=bool)
        self.INDEX_ULB = np.zeros([], dtype=bool)
        self.QUERIED_HISTORY = []
        self.DATA_INFOS = {'train': [], 'train_full': [], 'train_u': [], 'val': []}
        self.SUBSET = 10000
        self.ORI_SIZE = 0

        self.DATA_PATH = data_path
        self.load_data()
        self.num_samples = len(self.DATA_INFOS['train_full'])
        # if initial_size is None:
        #     initial_size = self.num_samples // 100
        if subset is None:
            self.SUBSET = max(10000, self.num_samples // 10)
        else:
            self.SUBSET = subset
        self.initialize_lb()

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def get_cat_ids(self, idx):
        return self.DATA_INFOS[idx]['gt_label'].astype(np.int)

    # Only used for loading data
    def prepare_data(self, idx, split, transform=None, aug_transform=None):
        raise NotImplementedError

    # Used for initialize data info
    def load_data(self):
        raise NotImplementedError

    # Add new labeled samples
    def update_lb(self, new_lb):
        self.INDEX_LB[new_lb] = True
        self.DATA_INFOS['train'] = list(np.array(self.DATA_INFOS['train_full'])[self.INDEX_LB])
        self.select_ulb()
        self.QUERIED_HISTORY.append(list(new_lb))

    def initialize_lb(self):
        self.DATA_INFOS['train'] = copy.deepcopy(self.DATA_INFOS['train_init'])
        self.select_ulb()

    def select_ulb(self):
        U_TEMP = np.arange(len(self.DATA_INFOS['train_full']))[~self.INDEX_LB]
        if self.SUBSET >= len(self.DATA_INFOS['train_full']) - len(self.DATA_INFOS['train']):
            U_SELECTED = U_TEMP
        else:
            U_SELECTED = np.random.choice(U_TEMP, self.SUBSET, replace=False)
        self.INDEX_ULB = np.array([True if x in U_SELECTED else False
                                   for x in range(len(self.DATA_INFOS['train_full']))])
        self.DATA_INFOS['train_u'] = np.array(self.DATA_INFOS['train_full'])[self.INDEX_ULB]
