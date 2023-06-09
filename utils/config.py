import argparse

__all__ = ['parse_commandline_args']


def create_parser():
    """Get the args from the command line"""
    parser = argparse.ArgumentParser(description='Deep active learning args --PyTorch ')

    # 与stable diffusion相关的配置
    parser.add_argument('--stable-diffusion-url', default="http://127.0.0.1:7860", type=str,
                        help='the url of stable diffusion')
    parser.add_argument('--categories', default=['sidewinder'], type=str,
                        nargs='+', help='categories to train, choose from '
                                        'axolotl, crampfish, emperor_penguin_chick, frilled_lizard, garfish, '
                                        'indian_cobra, king_penguin_chick, lycorma_delicatula, sidewinder, xylotrechus')

    # 与存储相关的必要信息
    parser.add_argument('--work-dir', default=None, type=str, help='the dir to save logs and models')
    parser.add_argument('--save-freq', default=100, type=int, metavar='EPOCHS',
                        help='checkpoint frequency(default: 100)')

    # 常规深度模型训练配置
    parser.add_argument('--dataset', type=str, default='MyImageFolder', metavar='DATASET',
                        help='The name of the used dataset(default: MyImageFolder)')
    parser.add_argument('--cls-load-path', type=str, default=None, help='which pth file to preload')
    parser.add_argument('--scoring-load-path', type=str, default=None, help='which pth file to preload')

    # 主动学习策略选择
    parser.add_argument('--strategy', type=str, default='ScoreBasedSampling',
                        help='which sampling strategy to choose')
    parser.add_argument('--n-cycle', default=5, type=int,
                        metavar='N', help='number of query rounds(default: 10)')
    parser.add_argument('--num-query', default=100, type=int,
                        metavar='N', help='number of query samples per epoch(default: 1000)')
    parser.add_argument('--subset', default=10000, type=int,
                        metavar='N', help='the size of the unlabeled pool to query, subsampling')
    parser.add_argument('--updating', action='store_true', help='Whether to use updating or retraining')

    # 分类模型训练配置
    parser.add_argument('--cls-n-epoch', default=5, type=int, metavar='N',
                        help='number of total training epochs(default: 100)')
    parser.add_argument('--cls-batch-size', type=int, default=64, metavar='BATCH_SIZE',
                        help='Batch size in both train and test phase(default: 64)')  # 16384
    parser.add_argument('--cls-num-workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--cls-optim-type', default='adam', type=str, metavar='CLS_OPTIM_NAME',
                        help='Type of optimizer')
    parser.add_argument('--cls-lr', '--learning-rate', default=3e-4, type=float, metavar='LR',
                        help='max learning rate (default: 0.1)')
    parser.add_argument('--cls-momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--cls-weight-decay', default=1e-4, type=float, help='weight decay (default: 0.0001)')

    # 打分模型训练配置
    parser.add_argument('--scoring-n-epoch', default=5, type=int, metavar='N',
                        help='number of total training epochs(default: 100)')
    parser.add_argument('--scoring-batch-size', type=int, default=256, metavar='BATCH_SIZE',
                        help='Batch size in both train and test phase(default: 64)')
    parser.add_argument('--scoring-num-workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--scoring-optim-type', default='adam', type=str, metavar='SCORE_OPTIM_NAME',
                        help='Type of optimizer')
    parser.add_argument('--scoring-lr', default=1e-3, type=float, metavar='LR',
                        help='max learning rate (default: 0.1)')
    parser.add_argument('--scoring-momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--scoring-weight-decay', default=1e-4, type=float, help='weight decay (default: 0.0001)')

    # 验证打分配置
    parser.add_argument('--validation-batch-size', default=64, type=int, metavar='N',
                        help='number of data loading workers (default: 64)')
    # 图像生成配置
    parser.add_argument('--initial-generated-images-per-class', default=1000, type=int,
                        help='number of images generated for training per class,'
                             'only available when there are no generated images in the dataset')
    parser.add_argument('--validation-generated-images-per-class', default=10, type=int,
                        help='number of images generated for validation per cycle')

    # Random Seed
    parser.add_argument('--seed', default=None, type=int, metavar='SEED', help='Random seed (default: None)')

    # 输出配置
    parser.add_argument('--out-iter-freq', default=1, type=int)

    # 与embedding和hypernetwork训练相关的配置
    parser.add_argument('--embedding-steps-per-lr', default=100, type=int)
    parser.add_argument('--save-embedding-every', default=5, type=int)
    parser.add_argument('--embedding-learn-rate', default=[5e-4, 2.5e-4, 7.5e-5, 5e-5, 2.5e-5], type=float, nargs='+',
                        help='learning rate to use in embedding training')
    parser.add_argument('--hypernetwork-steps-per-lr', default=100, type=int)
    parser.add_argument('--save-hypernetwork-every', default=5, type=int)
    parser.add_argument('--hypernetwork-learn-rate', default=[5e-6, 2.5e-6, 7.5e-7, 5e-7, 2.5e-7], type=float,
                        nargs='+', help='learning rate to use in hypernetwork training')

    return parser


def parse_commandline_args():
    """Returns the args from the command line"""
    return create_parser().parse_args()
