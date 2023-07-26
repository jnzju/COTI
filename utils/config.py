import argparse

__all__ = ['parse_commandline_args']


def create_parser():
    """Get the args from the command line"""
    parser = argparse.ArgumentParser(description='Deep active learning args --PyTorch ')

    ################## 更换环境之后需要更改的config 项目 #######################
    ''' 
    --server
    --stable-diffusion-model-path
    --dataset-path
    --memory-bank-path
    '''
    
    # server
    parser.add_argument('--server', default='gpu25', type=str,
                        help=
                        'gpu25'
                        'gpu_school')
    parser.add_argument('--task-path', default=None, type=str)
    parser.add_argument('--embed-task-path', default='/storage/home/lanzhenzhongLab2/yangjianan/yangjianan/zhangyanming/embed_tasks', type=str)
    parser.add_argument('--hyper-task-path', default='/storage/home/lanzhenzhongLab2/yangjianan/yangjianan/zhangyanming/hyper_tasks', type=str)
    parser.add_argument('--test-task-path', default='/storage/home/lanzhenzhongLab2/yangjianan/yangjianan/zhangyanming/test_tasks', type=str)
    parser.add_argument('--task-num', default=10, type=int)
    parser.add_argument('--category', default='axolotl', type=str, help=
                        'axolotl'
                        'crampfish'
                        'emperor_penguin_chick'
                        'indian_cobra'
                        'lycorma_delicatula'
                        )
    parser.add_argument('--class-idx',default=0, type=int)
    parser.add_argument('--subcategory', default='frilled_lizard', type=str, help=
                        'frilled_lizard'
                        'garfish'
                        'king_penguin_chick'
                        'sidewinder'
                        'xylotrechus'
                        )
    parser.add_argument('--subclass-idx', default=3, type=int)
    
    # 与stable diffusion相关的配置
    parser.add_argument('--stable-diffusion-url', default="http://127.0.0.1:7861", type=str,
                        help='the url of stable diffusion')
    parser.add_argument('--stable-diffusion-model-path', default="/storage/home/lanzhenzhongLab2/yangjianan/yangjianan/stable-diffusion-webui", type=str, 
                        help='example :  /storage/home/lanzhenzhongLab2/yangjianan/yangjianan/stable-diffusion-webui'
                                        '/home/yangjn/stable-diffusion-webui')

    # 与存储相关的必要信息
    parser.add_argument('--work-dir', default=None, type=str, help='the dir to save logs and models')
    parser.add_argument('--save-freq', default=100, type=int, metavar='EPOCHS',
                        help='checkpoint frequency(default: 100)')

    # 常规深度模型训练配置
    parser.add_argument('--dataset', type=str, default='MyImageFolder', metavar='DATASET',
                        help='The name of the used dataset(default: MyImageFolder)')
    parser.add_argument('--dataset-path', type=str, default="/storage/home/lanzhenzhongLab2/yangjianan/yangjianan/zhangyanming/data/stable_diffusion_dataset")
    parser.add_argument('--training-dataset',type=str, default='/storage/home/lanzhenzhongLab2/yangjianan/yangjianan/zhangyanming/data/stable_diffusion_dataset/training_dataset')
    parser.add_argument('--training-dataset-initial',type=str, default='/storage/home/lanzhenzhongLab2/yangjianan/yangjianan/zhangyanming/data/stable_diffusion_dataset/training_dataset_initial')
    parser.add_argument('--cls-load-path', type=str, default=None, help='which pth file to preload')
    parser.add_argument('--scoring-load-path', type=str, default=None, help='which pth file to preload')

    # 主动学习策略选择
    parser.add_argument('--strategy', type=str, default='ScoreBasedSampling',
                        help='which sampling strategy to choose')
    parser.add_argument('--updating', action='store_true', help='Whether to use updating or retraining')

    # 分类模型训练配置
    parser.add_argument('--cls-n-epoch', default=5, type=int, metavar='N',
                        help='number of total training epochs(default: 100), used: 5')
    parser.add_argument('--cls-batch-size', type=int, default=64, metavar='BATCH_SIZE',
                        help='Batch size in both train and test phase(default: 64), used: 64')  # 16384
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
                        help='number of total training epochs(default: 100), used 5')
    parser.add_argument('--scoring-batch-size', type=int, default=256, metavar='BATCH_SIZE',
                        help='Batch size in both train and test phase(default: 64), used: 256')
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
    parser.add_argument('--validation-generated-images-per-class', default=500, type=int,
                        help='number of images generated for validation per cycle, used: 10')

    # Random Seed
    parser.add_argument('--seed', default=None, type=int, metavar='SEED', help='Random seed (default: None)')

    # 输出配置
    parser.add_argument('--out-iter-freq', default=1, type=int)

    # 与embedding和hypernetwork训练相关的配置
    parser.add_argument('--sd-train-method', default='embed', type=str, help=
                        'embed'
                        'hyper'
                        'testembed'
                        'testhyper'
                        )
    
    parser.add_argument('--embedding-learn-rate', default=[5e-4, 2.5e-4, 7.5e-5, 5e-5, 2.5e-5], type=float, nargs='+',
                        help=
                        'learning rate to use in embedding training'
                        '[5e-4, 2.5e-4, 7.5e-5, 5e-5, 2.5e-5]'
                        )
    parser.add_argument('--hypernetwork-steps-per-lr', default=100, type=int)
    parser.add_argument('--save-hypernetwork-every', default=50, type=int)
    parser.add_argument('--hypernetwork-learn-rate', default=[5e-6, 2.5e-6, 7.5e-7, 5e-7, 2.5e-7], type=float,
                        nargs='+', help=
                        'learning rate to use in hypernetwork training'
                        '[5e-6, 2.5e-6, 7.5e-7, 5e-7, 2.5e-7]'
                        )
    
    # memory bank相关配置
    parser.add_argument('--mb-path', type=str, default=None,
                        help=   "/storage/home/lanzhenzhongLab2/yangjianan/yangjianan/zhangyanming/data/memorybank"
                                "/home/yangjn/zhangyanming/data/memorybank")
    parser.add_argument('--mb-path-embed', type=str, default='/storage/home/lanzhenzhongLab2/yangjianan/yangjianan/zhangyanming/data/embed_memorybank')
    parser.add_argument('--mb-path-hyper', type=str, default='/storage/home/lanzhenzhongLab2/yangjianan/yangjianan/zhangyanming/data/hyper_memorybank')
    parser.add_argument('--mb-path-test', type=str, default='/storage/home/lanzhenzhongLab2/yangjianan/yangjianan/zhangyanming/data/test_memorybank')
    parser.add_argument('--mb-save-num',type=int, default=200, help='every time select mb_save_num images to store')
    parser.add_argument('--mb-load-num',type=int, default=200, help='every time select mb_load_num images to load')
    parser.add_argument('--mb-size', type=int, default=400)

    # tag matching score相关配置
    parser.add_argument('--tag-matching-strategy', default='binary_classification', type=str, 
                        help='please choose between these strategies:'  
                        'binary_classification'                         #这个就是南哥实现的
                        'representation_distance'                       #这个是使用clip预训练模型进行图文匹配
                        '...//not implemented'
                        )
    # aesthetic score相关配置
    parser.add_argument('--aesthetic-strategy', default='being used now',type=str, 
                        help='please choose between these strategies:'
                        '...//not implemented'
                        )
    return parser


def parse_commandline_args():
    """Returns the args from the command line"""
    return create_parser().parse_args()
