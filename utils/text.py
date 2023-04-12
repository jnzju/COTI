import datetime
import os.path as osp
from collections import OrderedDict
from utils.in_out import dump

import torch


class TextLogger(object):
    """A Logger specially customized for `Strategy`.
       Apart from the basic logger provided by `Strategy`
       (The basic logger outputs information in consoles and `*.log`),
       it additionally records information in `*.json`,
       makes it easier to plot neccessary figures in result analysis.

       Args:
           model: (torch.nn.Module)
               The model used in the task. Just provides the architecture.
               This is just for recording the information of devices and memory.
               It does not matter what its parameters are.
           args: (dict)
               The necessary args provided by Class `Strategy` in `query_strategies/strategy.py` module.
               At least it should include the following parameters:

               - timestamp: The starting timestamp of the strategy
               - work_dir: The working directory of the current strategy
               - logger: The basic logger used by class `Strategy`

    """

    def __init__(self, model, args, logger):
        self.start_iter = 0
        self.model = model
        self.args = args
        self.logger = logger
        timestamp = args['timestamp']
        self.json_log_path = osp.join(args['work_dir'],
                                      f'{timestamp}.log.json')
        self.time_sec_tot = 0
        # 如果有什么meta数据，保存在此，记得dump一次
        # self._dump_log()

    def _get_max_memory(self):
        """Get the maximum memory the current model might occupy."""
        device = getattr(self.model, 'output_device', None)
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([mem / (1024 * 1024)],
                              dtype=torch.int,
                              device=device)
        return mem_mb.item()

    def _log_info(self, log_dict, iters_per_epoch=None, max_iters=None, iter_count=None, interval=None):
        """Log the current training/evaluation/testing information in `*.log` and console

        Args:
            log_dict (dict):
                This dict should record all the information obtained from class `Strategy`

                - mode: train/eval/test mode
                - lr(training phase only): learning rate
                - epoch(training phase only): current epoch
                - iter(training phase only): current iter in the epoch
                - time(training phase only): How much time the current iter consumed. This used for estimating eta(estimated time of arrival).
                - memory(training phase only): How much memory the current model used(only available when GPUs are used)
                If order args are included(including acc, loss or other evaluation metrics) in log_dict, they will be also recorded.

            iters_per_epoch (int):
                The number of iters in each epoch, used only in the training phase.
            max_iters (int):
                the total number of iters in the training phase, used for eta estimation in the training phase.
            iter_count (int):
                the rank of the current iter in the whole training phase, used for eta estimation in the training phase.
            interval (int):
                how many iters have been executed since the last time we perform logging,
                used for eta estimation in the training phase.

        """

        if log_dict['mode'] == 'train':
            if isinstance(log_dict['lr'], dict):
                lr_str = []
                for k, val in log_dict['lr'].items():
                    lr_str.append(f'lr_{k}: {val:.3e}')
                lr_str = ' '.join(lr_str)
            else:
                lr_str = f'lr: {log_dict["lr"]:.3e}'

            # by epoch: Epoch [4][100/1000]
            log_str = f'Epoch [{log_dict["epoch"]}]' \
                      f'[{log_dict["iter"]}/{iters_per_epoch}]\t'
            log_str += f'{lr_str}, '

            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * interval)
                time_sec_avg = self.time_sec_tot / (
                    iter_count - self.start_iter + 1)
                eta_sec = time_sec_avg * (max_iters - iter_count - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f'eta: {eta_str}, '
                log_str += f'time: {log_dict["time"]:.3f}, '
                # statistic memory
                if torch.cuda.is_available():
                    log_str += f'memory: {log_dict["memory"]}, '
        else:
            # val/test time
            # here 1000 is the length of the val dataloader
            # by epoch: Epoch[val] [4][1000]
            log_str = f'Epoch({log_dict["mode"]}) \t'

        log_items = []
        for name, val in log_dict.items():
            # 以下属性已经过特殊处理，不再需要输出
            # 其他属性都需要一并输出
            if name in [
                    'mode', 'Epoch', 'iter', 'lr', 'time',
                    'memory', 'epoch'
            ]:
                continue
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)

        self.logger.info(log_str)

    def _dump_log(self, log_dict):
        """Dump log in json format.
        This can also be used when we want to log something not exhibited in the console.

        Args:
            log_dict (dict):
                This dict should record all the information obtained from class `Strategy`.
                The same as `_log_info`.

        """
        json_log = OrderedDict()
        for k, v in log_dict.items():
            json_log[k] = self._round_float(v)
        # only append log at last line
        with open(self.json_log_path, 'a+') as f:
            dump(json_log, f, file_format='json')
            f.write('\n')

    def _round_float(self, items):
        """Rounding float data to 5 significant digits. In order to clearly show the log.

        Args:
            items (float, list[float]):
                Float number or a list of float numbers.

        """
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, 5)
        else:
            return items

    def log(self, log_dict, iters_per_epoch=None, max_iters=None, iter_count=None, interval=None):

        """Log the current training/evaluation/testing information in `*.log` and console,
           then dump log in json format.

        Args:
            log_dict (dict):
                This dict should record all the information obtained from class `Strategy`

                - mode: train/eval/test mode
                - lr(training phase only): learning rate
                - epoch(training phase only): current epoch
                - iter(training phase only): current iter in the epoch
                - time(training phase only): How much time the current iter consumed. This used for estimating eta(estimated time of arrival).

                If order args are included(including acc, loss or other evaluation metrics) in log_dict, they will be also recorded.

            iters_per_epoch (int):
                The number of iters in each epoch, used only in the training phase.
            max_iters (int):
                the total number of iters in the training phase, used for eta estimation in the training phase.
            iter_count (int):
                the rank of the current iter in the whole training phase, used for eta estimation in the training phase.
            interval (int):
                how many iters have been executed since the last time we perform logging,
                used for eta estimation in the training phase.

        """

        if torch.cuda.is_available():
            log_dict['memory'] = self._get_max_memory()

        self._log_info(log_dict, iters_per_epoch, max_iters, iter_count, interval)
        self._dump_log(log_dict)
        return log_dict
