import argparse

__all__ = ['parse_commandline_args']


def create_parser():
    """Get the args from the command line"""
    parser = argparse.ArgumentParser(description='Deep active learning args --PyTorch ')

    # 与存储相关的必要信息
    parser.add_argument('--work-dir', type=str, help='the directory where models and logs are saved')
    parser.add_argument('--save_dir', default=None, type=str,
                        help='the directory to save figures, when not specified, it is just the work dir')

    # 使用的基本模型和训练集
    parser.add_argument('--model', default='resnet18_cifar', metavar='MODEL',
                        help='The model to use.(default: resnet18_cifar)')
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='DATASET',
                        help='The name of the used dataset(default: cifar10)')

    # 作图配置
    parser.add_argument('--plot-mode', default='tsne', type=str)
    parser.add_argument('--plot-using-predefined', action='store_true')
    parser.add_argument('--plot-preload-path', default='pretrained/resnet18_cifar_cifar10_tsne_embedded.npy', type=str,
                        help='applied in train mode only')

    return parser


def parse_commandline_args():
    """Returns the args from the command line"""
    return create_parser().parse_args()