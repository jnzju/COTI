import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from torchvision.transforms import functional as F
import warnings
from architectures.clip import ClipClassifier, ClipAestheticPredictor, scoring_pt_state


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


"""
num_epochs : 100
batch_size : 16384
num_workers : 8

model:
    input_dim : 1024
    num_classes : 1000
    layers : 
        - 4096
        - 2048
    activation : "relu"
    norm_layer : "layer"
adam :
    lr : 3e-4
    weight_decay : 1e-4
sgd :
    lr : 1e-3
    momentum : 0.9
    weight_decay : 1e-4
"""


def get_initialized_cls_module(lr=3e-4, momentum=0.9, weight_decay=1e-4, optim_type='adam', T_max=None, **kwargs):
    clf = torch.nn.DataParallel(ClipClassifier(**kwargs))
    clf = clf.cuda()
    # clf.apply(weight_init)  # can not be used for vit
    if optim_type == 'sgd':
        optimizer = optim.SGD(clf.parameters(), lr=lr, momentum=momentum,
                              weight_decay=weight_decay)
    elif optim_type == 'adam':
        optimizer = optim.Adam(clf.parameters(), lr=lr)
    else:
        optimizer = optim.AdamW(clf.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    return clf, optimizer, scheduler


def get_initialized_score_module(lr, momentum, weight_decay, optim_type='adam'):
    scoring = ClipAestheticPredictor(768)
    scoring.load_state_dict(scoring_pt_state)
    scoring = torch.nn.DataParallel(scoring)
    scoring = scoring.cuda()
    if optim_type == 'sgd':
        optimizer = optim.SGD(scoring.parameters(), lr=lr, momentum=momentum,
                              weight_decay=weight_decay)
    elif optim_type == 'adam':
        optimizer = optim.Adam(scoring.parameters(), lr=lr)
    else:
        optimizer = optim.AdamW(scoring.parameters(), lr=lr)
    return scoring, optimizer


def get_images(c, n, indices_class, dataset):  # get random n images from class c
    indices = indices_class[c]
    if 0 < len(indices_class[c]) < n:
        indices = np.repeat(indices, n // len(indices_class[c]) + 1)
    elif len(indices_class[c]) == 0:
        warnings.warn(f"No samples in class {dataset.CLASSES[c]}!")
        return torch.zeros([0, *tuple(dataset.get_raw_data(0).shape)])

    idx_shuffle = np.random.permutation(indices)[:n]
    data = torch.stack([dataset.get_raw_data(idx, 'train') for idx in idx_shuffle])
    return data


class UnNormalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be denormalized.
        Returns:
            Tensor: Denormalized image.
        """
        t = tensor.clone()
        inv_mean = [-m/s for m, s in zip(self.mean, self.std)]
        inv_std = [1/s for s in self.std]
        return F.normalize(t, inv_mean, inv_std, self.inplace)


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


def match_loss(gw_syn, gw_real, dis_metric):
    dis = torch.tensor(0.0).cuda()

    if dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s' % dis_metric)

    return dis


def get_one_hot_label(labels=None, num_classes=10):
    if isinstance(labels, int):
        result = torch.zeros(num_classes).cuda()
        result[labels] = 1
        return result
    elif isinstance(labels, list) or isinstance(labels, tuple):
        labels = torch.tensor(labels).cuda()
    return torch.zeros(labels.shape[0],
                       num_classes).cuda().scatter_(1, labels.view(-1, 1), 1)


def soft_cross_entropy(pred, soft_targets):
    """A method for calculating cross entropy with soft targets"""
    logsoftmax = torch.nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


def print_class_object(obj, name, logger):
    for key, value in obj.__dict__.items():
        logger.info("CONFIG -- {} - {}: {}".format(name, key, value))
