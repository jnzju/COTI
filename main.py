import datetime
import os
import time
import uuid
import random

import torch
import numpy as np
from datasets.builder import DATASETS
from query_strategies.builder import STRATEGIES
from utils.config import parse_commandline_args
from utils.logger import get_logger
from utils.collect_env import collect_env
from utils.timer import Timer

from datasets.two_class_subset import TwoClassImageFolderSubset
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
    uid = str(uuid.uuid1().hex)[:8]
    if config.work_dir is None:
        config.work_dir = os.path.join('tasks',
                                       '{}_{}_{}_{}'.format(
                                           config.dataset, config.strategy,
                                           datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), uid))
    os.makedirs(config.work_dir, mode=0o777, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    config.timestamp = timestamp
    log_file = os.path.join(config.work_dir, f'{timestamp}.log')
    logger = get_logger(name='AL_for_stable_diffusion', log_file=log_file)
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    # set seed
    if config.seed is not None:
        set_seed(config.seed)  # To make the process deterministic

    # load dataset
    dataset = DATASETS.build(
        dict(type=config.dataset, data_path=config.dataset_path, subset=config.subset,
             initial_generated_images_per_class=config.initial_generated_images_per_class,
             url=config.stable_diffusion_url))

    if config.categories is None:
        selected_classes_info = [(idx, name) for idx, name in enumerate(dataset.CLASSES)]
    else:
        class_to_idx = {name: i for i, name in enumerate(dataset.CLASSES)}
        selected_classes_info = [(class_to_idx[name], name) for name in config.categories]

    for class_idx, class_name in selected_classes_info:
        sub_workdir = os.path.join(config.work_dir, class_name)
        sub_dataset = TwoClassImageFolderSubset(dataset, class_idx, dataset.SUB_CATEGORY[class_idx])
        # create initial embedding/hypernetwork
        # create_embedding(config.stable_diffusion_url, class_name, overwrite_old=True)
        # create_hypernetwork(config.stable_diffusion_url, class_name, overwrite_old=True)
        # First Round Training
        # temp_processed_path = sub_dataset.move_selected_images(config.stable_diffusion_url)
        pass
        # preprocess images
        # start experiment
        n_pool = len(sub_dataset.DATA_INFOS['train_full_main_category'])
        n_init = len(sub_dataset.DATA_INFOS['train_init_main_category'])
        logger.info('current category: {}'.format(class_name))
        logger.info('cardinality of initial labeled pool: {}'.format(n_init))
        logger.info('cardinality of initial candidate pool: {}'.format(n_pool))

        # load network
        strategy = STRATEGIES.build(dict(type=config.strategy,
                                         dataset=sub_dataset,
                                         args=config,
                                         logger=logger, timestamp=timestamp,
                                         work_dir=sub_workdir))

        # print info
        logger.info('Dataset: {}'.format(config.dataset + "_" + class_name))
        logger.info('Seed {}'.format(config.seed))
        logger.info('Strategy: {}'.format(type(strategy).__name__))

        if config.cls_load_path is not None:
            strategy.clf.load_state_dict(torch.load(config.cls_load_path))
            logger.info(f'Get pretrained classification parameters from {config.cls_load_path}')
        if config.scoring_load_path is not None:
            strategy.clf.load_state_dict(torch.load(config.scoring_load_path))
            logger.info(f'Get pretrained scoring parameters from {config.scoring_load_path}')

        strategy.run()

        # plot acc - label_num curve
        plot_curve_scores(sub_workdir,
                          strategy.num_labels_list,
                          [strategy.classifier_score_list, strategy.aesthetic_score_list,
                           strategy.total_score_list],
                          ['Tag Matching Score', 'Aesthetic Score', 'Comprehensive Score'])


if __name__ == '__main__':
    with Timer():
        config = parse_commandline_args()
        run(config)
