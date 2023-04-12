from .base_dataset import BaseDataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from .dataset_wrappers import RepeatDataset


class Handler(Dataset):
    def __init__(self, dataset: BaseDataset, split, transform=None, task='scoring'):
        self.dataset = dataset
        self.split = split
        self.transform = transform
        self.task = task

    def __getitem__(self, idx):
        return self.dataset.prepare_data(idx, self.split, self.transform, task=self.task)

    def __len__(self):
        return len(self.dataset.DATA_INFOS[self.split])


class AugHandler(Handler):
    def __init__(self, dataset: BaseDataset, split, transform=None, task='scoring'):
        super(AugHandler, self).__init__(dataset, split, transform, task)
        # just used for augmented part

    def __getitem__(self, idx):
        aug_data_info = self.dataset.DATA_INFOS[self.split]
        return self.dataset.prepare_data(aug_data_info[idx]['idx'], aug_data_info[idx]['split'],
                                         self.transform, aug_data_info[idx]['aug_transform'], self.task)


loader_dict = {'base': Handler, 'aug': AugHandler}


def GetHandler(dataset: BaseDataset, split: str, transform=None,
               repeat=None, loader_name=None, task='scoring'):
    # split can be either a string or a list of strings
    if type(split) == str:
        if loader_name is None:
            loader_name = 'base'
        h = loader_dict[loader_name](dataset, split, transform, task=task)
    elif type(split) in [list, tuple, set]:
        if repeat is None:
            repeat = [1 for _ in split]
        if loader_name is None:
            loader_name = ['base' for _ in split]
        assert len(split) == len(repeat)
        h_list = [RepeatDataset(loader_dict[loader_name](dataset, split_elem, transform, task=task), repeat_times)
                  for split_elem, repeat_times, loader_name in zip(split, repeat, loader_name)]
        h = ConcatDataset(h_list)
    else:
        raise Exception("Not supported type!")
    return h


def GetDataLoader(dataset: BaseDataset, split: str, transform=None, repeat=None, loader_name=None, task='scoring',
                  **kwargs):
    h = GetHandler(dataset, split, transform, repeat, loader_name, task)
    return DataLoader(h, **kwargs)
