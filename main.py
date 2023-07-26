import datetime
import os
import time
import uuid
import random

import torch
import numpy as np
from datasets.base_dataset import BaseDataset
from query_strategies.strategy import Strategy
from utils.config import parse_commandline_args
from utils.logger import get_logger
from utils.collect_env import collect_env
from utils.timer import Timer

from plot.curve import plot_curve_scores


def set_seed(seed=0):
    """If the seed is specified, the process will be deterministic.

    :param seed: the seed you wanna set
    :return: None

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子；
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，为所有的GPU设置种子。
    torch.backends.cudnn.deterministic = True  # CPU和GPU结果一致
    torch.backends.cudnn.benchmark = False


def run(config: dict = None):
    if config.sd_train_method == 'embed':
        config.task_path = config.embed_task_path
        config.mb_path = config.mb_path_embed
    elif config.sd_train_method == 'hyper':
        config.task_path = config.hyper_task_path
        config.mb_path = config.mb_path_hyper
    elif config.sd_train_method == 'testembed':
        config.sd_train_method == 'embed'
        config.task_path = config.test_task_path
        config.mb_path = config.mb_path_test
    elif config.sd_train_method == 'testhyper':
        config.sd_train_method == 'hyper'
        config.task_path = config.test_task_path
        config.mb_path = config.mb_path_test
    else :
        raise NotImplementedError
    
    if not os.path.exists(config.task_path):
        os.makedirs(config.task_path, mode=0o777, exist_ok=True)
    
    uid = str(uuid.uuid1().hex)[:8]
    
    config.work_dir = os.path.join(config.task_path, '{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), uid))
    
    os.makedirs(config.work_dir, mode=0o777, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    config.timestamp = timestamp
    log_file = os.path.join(config.work_dir, 'log')
    logger = get_logger(name='AL_for_stable_diffusion', log_file=log_file)
    
    # set seed
    if config.seed is not None:
        set_seed(config.seed)  # To make the process deterministic
    
    # load dataset
    dataset = BaseDataset(config)
    
    n_pool = len(dataset.DATA_INFOS['train_full'])
    n_init = len(dataset.DATA_INFOS['train_init'])
    logger.info('current category: {}'.format(config.category))
    logger.info('cardinality of initial labeled pool: {}'.format(n_init))
    logger.info('cardinality of initial candidate pool: {}'.format(n_pool))

    # load network
    strategy = Strategy(dataset=dataset, args=config, logger=logger)

    strategy.run()

    # plot acc - label_num curve
    plot_curve_scores(config.work_dir,
                        strategy.num_labels_list,
                        [strategy.classifier_score_list, strategy.aesthetic_score_list,
                        strategy.total_score_list],
                        ['Tag Matching Score', 'Aesthetic Score', 'Comprehensive Score'])


if __name__ == '__main__':
    with Timer():
        config = parse_commandline_args()
        run(config)
