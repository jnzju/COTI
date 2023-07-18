import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import os
from getpass import getuser
from socket import gethostname
from utils.progressbar import track_iter_progress
from utils.text import TextLogger
from utils.timer import Timer
from .builder import STRATEGIES
from datasets.dataloader import GetDataLoader
from datasets.base_dataset import BaseDataset
from datasets.generated_dataset import generate_virtual_dataset
from evaluation import *
from .utils import get_initialized_cls_module, get_initialized_score_module, get_lr
from architectures.clip import get_clip_image_features
from embedding.embedding import create_embedding, embedding_training
from embedding.hypernetwork import create_hypernetwork, hypernetwork_training
import math
import csv
from evaluation import inception_score, IgnoreLabelDataset
from pytorch_fid.fid_score import calculate_fid_given_paths
from PIL import ImageFile
import shutil
ImageFile.LOAD_TRUNCATED_IMAGES = True


@STRATEGIES.register_module()
class Strategy:
    def __init__(self, dataset: BaseDataset, args, logger, timestamp, work_dir):
        self.dataset = dataset
        self.args = args
        self.work_dir = work_dir
        
        # Model
        self.init_models()
        
        # This is for resume
        self.cycle = 0
        self.cls_epoch = 0
        self.scoring_epoch = 0
        self.logger = logger
        self.TextLogger = TextLogger(self.cls_net, vars(args), logger)
        self.timer = Timer()
        self.timestamp = timestamp
        self.classifier_score_list = []
        self.aesthetic_score_list = []
        self.total_score_list = []
        self.fid_score_list = []
        self.is_score_list = []
        self.r_precision_list = []
        self.num_labels_list = []
        self.TextLogger._dump_log(vars(args))
        self.pre_score()
        
        # use memory bank
        self.mb_setup(
            data_path=os.path.join(self.dataset.DATA_PATH, "training_dataset"),
            category=self.dataset.CLASSES[0],
            gt_label=0,
            mb_load_num=self.args.mb_load_num,
            mb_save_num=self.args.mb_save_num,
            mb_path=self.args.memory_bank_path
        )
        self.merge_mb(split='train')

    def init_models(self):
        """When we want to initialize the model we use, apply this function.
        Random parameter initialization is included.
        """
        self.cls_net, self.cls_optimizer, self.cls_scheduler = \
            get_initialized_cls_module(self.args.cls_lr, self.args.cls_momentum,
                                       self.args.cls_weight_decay,
                                       self.args.cls_optim_type,
                                       self.args.cls_n_epoch * math.ceil(
                                           len(self.dataset.DATA_INFOS['train']) / self.args.cls_batch_size),
                                       num_classes=len(self.dataset.CLASSES))
        self.scoring_net, self.scoring_optimizer = \
            get_initialized_score_module(self.args.scoring_lr, self.args.scoring_momentum,
                                         self.args.scoring_weight_decay)

    def pre_score(self, split=None):
        if split is None:
            for temp_split in self.dataset.DATA_INFOS.keys():
                if temp_split in ['train', 'train_full', 'train_u', 'val', 'train_generated']:
                    continue
                if temp_split not in ['train_init_main_category']:
                    score_list = self.predict(self.scoring_net, split=temp_split, metric='aesthetic_score')
                    for i in range(len(self.dataset.DATA_INFOS[temp_split])):
                        self.dataset.DATA_INFOS[temp_split][i]['aesthetic_score'] = score_list[i].item()
                else:
                    for i in range(len(self.dataset.DATA_INFOS[temp_split])):
                        self.dataset.DATA_INFOS[temp_split][i]['aesthetic_score'] = 10.0
        else:
            score_list = self.predict(self.scoring_net, split=split, metric='aesthetic_score')
            for i in range(len(self.dataset.DATA_INFOS[split])):
                self.dataset.DATA_INFOS[split][i]['aesthetic_score'] = score_list[i].item()
        self.dataset.initialize_lb()

    def query(self, n):
        """Query new samples according the current model or some strategies.

        :param n: (int)The number of samples to query.

        Returns:
            list[int]: The indices of queried samples.

        """
        raise NotImplementedError

    def update(self, n):
        idxs_q = self.query(n)
        for idx_q in idxs_q:
            pre_score = self.dataset.DATA_INFOS['train_full_main_category'][int(idx_q)]['aesthetic_score']
            self.dataset.DATA_INFOS['train_full_main_category'][int(idx_q)]['aesthetic_score'] = \
                10.0 - (10.0 - pre_score) / self.args.n_cycle * self.cycle
        self.dataset.update_lb(idxs_q)

    def _classifier_train(self, loader_tr, clf_group: dict, clf_name='train_cls', log_show=True):
        """Represents one epoch.

        :param loader_tr: (:obj:`torch.utils.data.DataLoader`) The training data wrapped in DataLoader.

        Accuracy and loss in the each iter will be recorded.

        """
        iter_out = self.args.out_iter_freq
        loss_list = []
        right_count_list = []
        samples_per_batch = []
        clf_group['clf'].train()

        for batch_idx, (x, y, no, idx, _) in enumerate(loader_tr):
            x, y = torch.squeeze(x, 1).cuda(), y.cuda()
            if x.shape[0] <= 1:
                break
            clf_group['optimizer'].zero_grad()  #模型参数梯度初始化为0
            out = clf_group['clf'](x)   # 前向传播计算预测值
            pred = out.max(1)[1]
            right_count_list.append((pred == y).sum().item())
            samples_per_batch.append(len(y))
            loss = F.cross_entropy(out, y)  # 计算当前损失

            loss_list.append(loss.item())
            loss.backward()     #反向传播计算梯度
            clf_group['optimizer'].step() # 更新所有的参数
            iter_time = self.timer.since_last_check()
            if log_show:
                if (batch_idx + 1) % iter_out == 0:
                    log_dict = dict(
                        mode=clf_name,  # 训练模式
                        epoch=self.cls_epoch,  # 当前是第几个epoch
                        iter=batch_idx + 1,  # 当前是第几个iter
                        lr=get_lr(clf_group['optimizer']),  # 获取当前optimizer的学习率
                        time=iter_time,  # 当前iter消耗的时间
                        acc=1.0 * np.sum(right_count_list[-iter_out:]) / np.sum(samples_per_batch[-iter_out:]),
                        loss=np.sum(loss_list[-iter_out:])
                    )
                    self.TextLogger.log(
                        log_dict=log_dict,
                        iters_per_epoch=len(loader_tr),
                        iter_count=self.cls_epoch * len(loader_tr) + batch_idx,
                        max_iters=self.args.cls_n_epoch * len(loader_tr),  # 一个active round总共的epoch数量
                        interval=iter_out  # 多少个iter进行一次log
                    )
        clf_group['scheduler'].step()

    def _scorer_train(self, loader_tr, clf_group: dict, clf_name='train_scoring', log_show=True):
        """Represents one epoch.

                :param loader_tr: (:obj:`torch.utils.data.DataLoader`) The training data wrapped in DataLoader.

                Accuracy and loss in the each iter will be recorded.

                """
        iter_out = self.args.out_iter_freq
        loss_list = []
        samples_per_batch = []
        clf_group['scoring'].train()
        for batch_idx, (x, _, no, idx, score_y) in enumerate(loader_tr):
            x, score_y = x.cuda(), score_y.to(torch.float).cuda()
            clf_group['optimizer'].zero_grad()
            out = torch.flatten(clf_group['scoring'](x))
            samples_per_batch.append(len(score_y))
            loss = F.mse_loss(out, score_y)

            loss_list.append(loss.item())
            loss.backward()
            clf_group['optimizer'].step()
            iter_time = self.timer.since_last_check()

            if log_show:
                if (batch_idx + 1) % iter_out == 0:
                    log_dict = dict(
                        mode=clf_name,  # 训练模式
                        epoch=self.scoring_epoch,  # 当前是第几个epoch
                        iter=batch_idx + 1,  # 当前是第几个iter
                        lr=get_lr(clf_group['optimizer']),  # 获取当前optimizer的学习率
                        time=iter_time,  # 当前iter消耗的时间
                        loss=np.sum(loss_list[-iter_out:])
                    )
                    self.TextLogger.log(
                        log_dict=log_dict,
                        iters_per_epoch=len(loader_tr),
                        iter_count=self.scoring_epoch * len(loader_tr) + batch_idx,
                        max_iters=self.args.scoring_n_epoch * len(loader_tr),  # 一个active round总共的epoch数量
                        interval=iter_out  # 多少个iter进行一次log
                    )

    def train(self):
        self.logger.info('Start running, host: %s, work_dir: %s',
                         f'{getuser()}@{gethostname()}', self.args.work_dir)
        self.logger.info('max: %d classification epochs', self.args.cls_n_epoch)
        self.logger.info('max: %d scoring epochs', self.args.scoring_n_epoch)
        self.cls_net.train()
        self.scoring_net.train()

        loader_tr_cls = GetDataLoader(self.dataset, split='train', shuffle=True, batch_size=self.args.cls_batch_size,
                                      num_workers=self.args.cls_num_workers, task='cls')
        loader_tr_scoring = GetDataLoader(self.dataset, split='train', shuffle=True,
                                          batch_size=self.args.cls_batch_size,
                                          num_workers=self.args.scoring_num_workers, task='scoring')
        while self.cls_epoch < self.args.cls_n_epoch:
            self.timer.since_last_check()
            self._classifier_train(loader_tr_cls,
                                   {'clf': self.cls_net, 'optimizer': self.cls_optimizer, 'scheduler': self.cls_scheduler})
            self.cls_epoch += 1
        while self.scoring_epoch < self.args.scoring_n_epoch:
            self.timer.since_last_check()
            self._scorer_train(loader_tr_scoring,
                               {'scoring': self.scoring_net, 'optimizer': self.scoring_optimizer})
            self.scoring_epoch += 1

        self.cls_epoch = 0
        self.scoring_epoch = 0
        # self.save()
        self.cls_net.eval()
        self.scoring_net.eval()

    def _embedding_train(self, cycle, temp_processed_path):
        embedding_iters = 0
        create_embedding(self.args.stable_diffusion_url, self.args.stable_diffusion_model_path, self.dataset.CLASSES[0], overwrite_old=True)
        os.makedirs(os.path.join(os.path.abspath('.'),
                                 self.work_dir, f'active_round_{cycle}', "embedding"), mode=0o777, exist_ok=True)
        
        first_flag=True
        for lr in self.args.embedding_learn_rate:
            self.logger.info(f"Training with learning rate {lr} at cycle {cycle}!")
            embedding_path_list, image_path_list = \
                embedding_training(self.args.server,
                                   self.args.stable_diffusion_model_path,
                                   self.args.stable_diffusion_url,
                                   self.dataset.CLASSES[0],
                                   learn_rate=lr,
                                   data_root=temp_processed_path,
                                   log_directory=os.path.join(os.path.abspath('.'),
                                                              self.work_dir, f'active_round_{cycle}', "embedding"),
                                   steps=embedding_iters + self.args.embedding_steps_per_lr,
                                   initial_step=embedding_iters,
                                   save_embedding_every=self.args.save_embedding_every,
                                   template_filename="style_filewords.txt",
                                   preview_prompt=f"a_photo_of_{self.dataset.CLASSES[0]}, "
                                                  f"{self.dataset.CLASSES[0]}, real_life")
            # 接下来筛选最优embedding
            self.dataset.DATA_INFOS['temp'] = [{'no': i, 'img': path, 'gt_label': 0, 'aesthetic_score': 0.
                                                } for i, path in enumerate(image_path_list)]
            aesthetic_score_list = self.predict(self.scoring_net, split='temp', metric='aesthetic_score')
            tag_matching_score_list = self.predict(self.cls_net, split='temp', metric='tag_matching_score')
            embedding_iters = embedding_iters + self.args.embedding_steps_per_lr
            total_score_list = aesthetic_score_list + tag_matching_score_list
            best_idx = torch.argmax(total_score_list).item()
            if first_flag:
                best_score = total_score_list[best_idx]
                first_flag=False
            elif total_score_list[best_idx] >= best_score:
                best_score=total_score_list[best_idx]
            
                # 选择完毕，移动最佳embedding替换，删除多余的embedding
                embedding_pt_dict = torch.load(embedding_path_list[best_idx])
                embedding_pt_dict['name'] = self.dataset.CLASSES[0]
                dsc_file = os.path.join(self.args.stable_diffusion_model_path, "embeddings", f"{self.dataset.CLASSES[0]}.pt")
                os.remove(dsc_file)
                # shutil.copy(embedding_path_list[best_idx], dsc_file)
                torch.save(embedding_pt_dict, dsc_file)
                """
                for del_idx in range(best_idx + 1, self.args.embedding_steps_per_lr // self.args.save_embedding_every):
                    os.remove(image_path_list[del_idx])
                    os.remove(embedding_path_list[del_idx])
                """
            del self.dataset.DATA_INFOS['temp']

    def _hypernetwork_train(self, cycle, temp_processed_path):
        hypernetwork_iters = 0
        create_hypernetwork(self.args.stable_diffusion_url, self.args.stable_diffusion_model_path, self.dataset.CLASSES[0], overwrite_old=True)
        os.makedirs(os.path.join(os.path.abspath('.'),
                                 self.work_dir, f'active_round_{cycle}', "hypernetwork"), mode=0o777, exist_ok=True)
        for lr in self.args.hypernetwork_learn_rate:
            hypernetwork_path_list, image_path_list = \
                hypernetwork_training(self.args.stable_diffusion_url,
                                      self.dataset.CLASSES[0],
                                      learn_rate=lr,
                                      data_root=temp_processed_path,
                                      log_directory=os.path.join(os.path.abspath('.'),
                                                                 self.work_dir, f'active_round_{cycle}', "hypernetwork"),
                                      steps=hypernetwork_iters + self.args.hypernetwork_steps_per_lr,
                                      initial_step=hypernetwork_iters,
                                      save_hypernetwork_every=self.args.save_hypernetwork_every,
                                      template_file=os.path.join(os.path.abspath(".."),
                                                                 "stable-diffusion-webui",
                                                                 "textual_inversion_templates", "hypernetwork.txt"),
                                      preview_prompt=f"a_photo_of_{self.dataset.CLASSES[0]}, "
                                                     f"{self.dataset.CLASSES[0]}, real_life")

            # 接下来筛选最优hypernetwork
            self.dataset.DATA_INFOS['temp'] = [{'no': i, 'img': path, 'gt_label': 0, 'aesthetic_score': 0.
                                                } for i, path in enumerate(image_path_list)]
            aesthetic_score_list = self.predict(self.scoring_net, split='temp', metric='aesthetic_score')
            tag_matching_score_list = self.predict(self.cls_net, split='temp', metric='tag_matching_score')
            total_score_list = aesthetic_score_list + tag_matching_score_list
            best_idx = torch.argmax(total_score_list).item()
            hypernetwork_iters = hypernetwork_iters + self.args.hypernetwork_steps_per_lr
            # 选择完毕，移动最佳embedding替换，删除多余的embedding
            hypernetwork_pt_dict = torch.load(hypernetwork_path_list[best_idx])
            hypernetwork_pt_dict['name'] = self.dataset.CLASSES[0]
            dsc_file = os.path.join(self.args.stable_diffusion_model_path, "models", "hypernetworks", f"{self.dataset.CLASSES[0]}.pt")
            os.remove(dsc_file)
            # shutil.copy(hypernetwork_path_list[best_idx], dsc_file)
            torch.save(hypernetwork_pt_dict, dsc_file)
            """
            for del_idx in range(best_idx + 1, self.args.hypernetwork_steps_per_lr // self.args.save_hypernetwork_every):
                os.remove(image_path_list[del_idx])
                os.remove(hypernetwork_path_list[del_idx])
            """
            del self.dataset.DATA_INFOS['temp']

    def embedding_train_cycle(self, cycle):
        temp_path = os.path.join(os.path.abspath('.'), self.work_dir, f'active_round_{cycle}', 'selected_images')
        temp_processed_path = self.dataset.move_selected_images(self.args.stable_diffusion_url, temp_path)
        self._embedding_train(cycle, temp_processed_path)
        # self._hypernetwork_train(cycle, temp_processed_path)

    def run(self):
        # Initial Embedding Training
        self.embedding_train_cycle('init')
        while self.cycle < self.args.n_cycle:
            active_path = os.path.join(self.work_dir, f'active_round_{self.cycle}')
            os.makedirs(active_path, mode=0o777, exist_ok=True)
            num_labels = len(self.dataset.DATA_INFOS['train_init_main_category']) + np.sum(self.dataset.INDEX_LB)
            self.logger.info(f'Active Round {self.cycle} with {num_labels} labeled instances')
            if self.cycle == 0:
                active_meta_log_dict = dict(
                    mode='active_meta',
                    cycle=self.cycle,
                    num_labels=num_labels,
                    idxs_lb=list(np.arange(len(self.dataset.DATA_INFOS['train_full_main_category']))[self.dataset.INDEX_LB])
                )
            else:
                active_meta_log_dict = dict(
                    mode='active_meta',
                    cycle=self.cycle,
                    num_labels=num_labels,
                    idxs_queried=list(self.dataset.QUERIED_HISTORY[-1]),
                    idxs_lb=list(np.arange(len(self.dataset.DATA_INFOS['train_full_main_category']))[self.dataset.INDEX_LB])
                )
            self.TextLogger._dump_log(active_meta_log_dict)
            self.init_models()
            self.train()
            self.update(self.args.num_query)  # Update the labeled pool according to the current model

            self.num_labels_list.append(num_labels)
            self.embedding_train_cycle(self.cycle)

            self.regenerate_validation_set()
            aesthetic_score_all = self.predict(self.scoring_net, split='val', metric='aesthetic_score')
            aesthetic_score_all[aesthetic_score_all > 10.0] = 10.0
            self.aesthetic_score_list.append(torch.mean(aesthetic_score_all).item())
            classifier_score_all = self.predict(self.cls_net, split='val', metric='tag_matching_score')
            self.classifier_score_list.append(torch.mean(classifier_score_all).item())
            self.total_score_list.append(10 * math.sin(self.aesthetic_score_list[-1] * 3.14 / 20)
                                         * math.sin(self.classifier_score_list[-1] * 3.14 / 20))

            self.r_precision_list.append(torch.mean(classifier_score_all).item())
            is_score_all = self.predict(self.scoring_net, split='val', metric='is')
            self.is_score_list.append(torch.mean(is_score_all).item())
            self.fid_score_list.append(self.predict(self.scoring_net, split='val', metric='fid'))

            log_dict = dict(mode='val', cycle=self.cycle,
                            aesthetic=self.aesthetic_score_list[-1],
                            total=self.total_score_list[-1],
                            r_precision=self.r_precision_list[-1],
                            is_score=self.is_score_list[-1],
                            fid_score=self.fid_score_list[-1])
            self.TextLogger.log(log_dict)

            self.cycle += 1

        self.record_evaluation_results()
        self.mb_store(
            category=self.dataset.CLASSES[0], 
            split='train'
        )

    def predict(self, clf, split='train', metric='accuracy',
                topk=None, n_drop=None, thrs=None, dropout_split=False, log_show=True):
        # For both evaluation and informative metric based on probabilistic outputs
        # Allowed split: train, train_full, val, test
        # Allowed metrics: accuracy, precision, recall, f1_score, support
        # The above metrics return a scalar
        # Allowed informative metrics: entropy, lc, margin
        # The above metrics return a vector of length N(The number of data points)
        # If in dropout split mode, The above metrics return a tensor of size [n_drop, N, C]
        if isinstance(clf, torch.nn.Module):
            clf.eval()
        if n_drop is None:
            n_drop = 1
        if topk is None:
            topk = 1
        if thrs is None:
            thrs = 0.
        # Predicting classification model quality
        if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'support']:
            loader = GetDataLoader(self.dataset, split=split,
                                   shuffle=False,
                                   batch_size=self.args.validation_batch_size, task='cls')
            # Evaluation Metric
            self.logger.info(f"Calculating Performance with {metric} on {split}...")
            pred = torch.zeros([len(self.dataset.DATA_INFOS[split]),
                                len(self.dataset.CLASSES)]).cuda()
            target = torch.zeros(len(self.dataset.DATA_INFOS[split]), dtype=torch.long).cuda()
            with torch.no_grad():
                for x, y, _, idxs, _ in track_iter_progress(loader):
                    x, y = x.cuda(), y.cuda()
                    if isinstance(clf, torch.nn.Module):
                        out, _, _ = clf(x)
                    else:
                        out = clf(x)
                    prob = F.softmax(out, dim=1)
                    pred[idxs] = prob
                    target[idxs] = y
            if metric == 'accuracy':
                result = accuracy(pred, target, topk, thrs)
            elif metric == 'precision':
                result = precision(pred, target, thrs=thrs)
            elif metric == 'recall':
                result = recall(pred, target, thrs=thrs)
            elif metric == 'f1_score':
                result = f1_score(pred, target, thrs=thrs)
            elif metric == 'support':
                result = support(pred, target)
            else:
                raise Exception(f"Metric {metric} not implemented!")
            if len(result) == 1:
                result = result.item()
            else:
                result = result.numpy().tolist()
            if log_show:
                log_dict = dict(mode=split, cycle=self.cycle)
                log_dict[metric] = result
                self.TextLogger.log(log_dict)

        elif metric in ['aesthetic_score', 'tag_matching_score', 'is', 'fid']:
            loader = GetDataLoader(self.dataset, split=split,
                                   shuffle=False,
                                   batch_size=self.args.validation_batch_size, task='scoring')
            self.logger.info(f"Calculating Informativeness with {metric} on {split}...")
            # 求图像得分的平均值
            # calculating scores
            if metric == 'aesthetic_score':
                result = []
                with torch.no_grad():
                    for x, _, _, _, _ in track_iter_progress(loader):
                        x = x.cuda()
                        x_scores = clf(x)
                        result.append(x_scores)
                    """
                    for data_dict in track_iter_progress(self.dataset.DATA_INFOS[split]):
                        pil_image = Image.open(data_dict['img'])
                        image_feature = get_clip_image_features(pil_image).to(torch.float)
                        prediction = clf(image_feature)
                        result.append(float(prediction))
                    """
                if len(result) > 0:
                    result = torch.flatten(torch.cat(result).cuda())
                else:
                    result = torch.tensor([]).cuda()
            elif metric == 'tag_matching_score':
                if self.args.tag_matching_strategy == 'binary_classification':
                    pred = torch.zeros([len(self.dataset.DATA_INFOS[split]),
                                        len(self.dataset.CLASSES)]).cuda()
                    y_list = []

                    with torch.no_grad():
                        for x, _, _, idxs, _ in track_iter_progress(loader):
                            x = torch.squeeze(x, 1).cuda()
                            out = clf(x)
                            pred[idxs] += F.softmax(out, dim=1)
                    result = torch.flatten(pred[:, 0] * 10.0)
                elif self.args.tag_matching_strategy == 'representation_distance':
                    pred = torch.zeros([len(self.dataset.DATA_INFOS[split]), len(self.dataset.CLASSES)]).cuda()
                    with torch.no_grad():
                        for _, _, _, idxs, _ in track_iter_progress(loader):
                            num = len(idxs)
                            image_paths = [self.dataset.DATA_INFOS[split][idxs[i]]['img'] for i in range(num)]
                            positive_text_input = "a " + self.dataset.CLASSES[0]
                            sub_text_input = "a " + self.dataset.father_dataset.CLASSES[self.dataset.sub_class_idx]
                            device = "cuda"
                            model, preprocess = clip.load("ViT-B/32", device=device)
                            
                            for i in range(num):
                                image = preprocess(Image.open(images_paths[i])).unsqueeze(0).to(device)
                                text = clip.tokenize([positive_text_input, sub_text_input]).to(device)
                                logits_per_image, logits_per_text = model(image, text)
                                pred[idxs[i]] = logits_per_image.softmax(dim=-1)
                    result = torch.flatten(pred[:, 0] * 10.0)
                else :
                    raise NotImplementedError
            elif metric == 'r_precision':
                raise NotImplementedError
            elif metric == 'is':
                raise NotImplementedError
            elif metric == 'fid':
                raise NotImplementedError
            else:
                raise NotImplementedError

        else:  # Informative Metric
            loader = GetDataLoader(self.dataset, split=split,
                                   shuffle=False,
                                   batch_size=self.args.validation_batch_size, task='cls')
            self.logger.info(f"Calculating Informativeness with {metric} on {split}...")
            if isinstance(clf, torch.nn.Module):
                clf.train()
            if dropout_split is False:
                pred = torch.zeros([len(self.dataset.DATA_INFOS[split]),
                                    len(self.dataset.CLASSES)]).cuda()
                for i in range(n_drop):
                    self.logger.info('n_drop {}/{}'.format(i + 1, n_drop))
                    with torch.no_grad():
                        for x, _, _, idxs, _ in track_iter_progress(loader):
                            x = torch.squeeze(x, 1).cuda()
                            out = clf(x)
                            pred[idxs] += F.softmax(out, dim=1)
                    # print(pred)
                pred /= n_drop
                if metric == 'entropy':
                    log_pred = torch.log(pred)
                    # the larger the more uncertain
                    result = - (pred * log_pred).sum(1)
                elif metric == 'lc':
                    # the smaller the more uncertain
                    result = pred.max(1)[0]
                elif metric == 'margin':
                    # the smaller the more uncertain
                    pred_sorted, _ = pred.sort(descending=True)
                    result = pred_sorted[:, 0] - pred_sorted[:, 1]
                elif metric == 'prob':
                    result = pred
                else:
                    raise Exception(f"Metric {metric} not implemented!")
            else:
                print("No metric will be used in dropout split mode!")
                result = torch.zeros([n_drop, len(self.dataset.DATA_INFOS[split]),
                                      len(self.dataset.CLASSES)]).cuda()
                for i in range(n_drop):
                    self.logger.info('n_drop {}/{}'.format(i + 1, n_drop))
                    with torch.no_grad():
                        for x, _, _, idxs, _ in track_iter_progress(loader):
                            x = get_clip_image_features(x).cuda()
                            out = clf(x)
                            result[i][idxs] += F.softmax(out, dim=1)
        # n_drops ignored
        return result
        # back to train split as the default split

    def save(self):
        """Save the current model parameters."""
        model_out_path = Path(os.path.join(self.args['work_dir'], f'active_round_{self.active_round}'))
        state = self.clf.state_dict(),
        if not model_out_path.exists():
            model_out_path.mkdir()
        save_target = model_out_path / f"active_round_{self.active_round}-" \
                                       f"label_num_{np.sum(self.idxs_lb).item()}-epoch_{self.epoch}.pth"
        torch.save(state, save_target)

        self.logger.info('==> save model to {}'.format(save_target))

    def regenerate_validation_set(self):
        self.dataset.DATA_INFOS['val'] = generate_virtual_dataset(
            url=self.args.stable_diffusion_url,
            prompt=f"a_photo_of_{self.dataset.CLASSES[0]}, " + self.dataset.CLASSES[0] + ", real_life",
            num_samples=self.args.validation_generated_images_per_class,
            temp_dir=os.path.join(self.work_dir, f"active_round_{self.cycle}", "temp_val_images"))

    def record_evaluation_results(self):
        file_name = os.path.join(self.args.work_dir, 'evaluation.csv')
        header = ['num_labels', 'aesthetic', 'total', 'is', 'fid', 'r_precision']
        with open(file_name, 'w', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(header)
            for i, cycle_recorder in enumerate(zip(self.aesthetic_score_list,
                                                   self.total_score_list,
                                                   self.is_score_list,
                                                   self.fid_score_list,
                                                   self.classifier_score_list)):
                f_csv.writerow([(i + 1) * self.args.num_query,
                                self.aesthetic_score_list[i],
                                self.total_score_list[i],
                                self.is_score_list[i],
                                self.fid_score_list[i],
                                self.classifier_score_list[i]])

    #####################  memory bank mechanism realization  ########################
    def mb_setup(self, data_path, category, gt_label, mb_load_num, mb_save_num, mb_path=None): #data_path是指到training_dataset
        self.mb_path = mb_path
        self.mb_load_num = mb_load_num
        self.mb_save_num = mb_save_num
        if not os.path.exists(self.mb_path) or self.mb_path==None:
            print('[Memory Bank Path ERROR] Please Specify Your Memory Bank Path!')
            exit(1)
        
        if os.path.exists(os.path.join(self.mb_path, category, 'img_score_match_list.npz')):
            self.mb_load(data_path=data_path, category=category, gt_label=gt_label)
        elif os.path.exists(os.path.join(self.mb_path, category)):
            print('memorybank/{category} being used is damaged, COTI needs to regenerate it and remove previous image data.')
            shutil.rmtree(os.path.join(self.mb_path, category))
            os.makedirs(os.path.join(self.mb_path, category), exist_ok=True)
        else:
            os.makedirs(os.path.join(self.mb_path, category), exist_ok=True)
        
        
    def mb_store(self, category, split='train'): # select top mb_selected_num images and save them to the memory bank
        split_length = len(self.dataset.DATA_INFOS[split])
        idxs = np.arange(split_length)
        aesthetic_scores=self.predict(self.scoring_net, split=split, metric='aesthetic_score')
        classifier_scores=self.predict(self.cls_net, split=split, metric='tag_matching_score')
        total_scores = (aesthetic_scores + classifier_scores).sort()[1].cpu().numpy() #[1]表示的是使用返回值的索引
        idxs_tostore = idxs[total_scores[-self.mb_save_num:]]
        
        aesthetic_scores = aesthetic_scores.cpu().numpy()
        classifier_scores = classifier_scores.cpu().numpy()
        
        store_path = os.path.join(self.mb_path, category)
        img_score_match_list = []
        if not os.path.exists(store_path):
            os.makedirs(store_path, mode=0o777, exist_ok=True)
        else:
            _lst = np.load(os.path.join(store_path, 'img_score_match_list.npz'), allow_pickle=True)
            arr_0 = _lst['arr_0']
            for i in arr_0:
                img_score_match_list.append(i)
        
        for i in idxs_tostore:
            no = self.dataset.DATA_INFOS[split][i]['no']
            image_path = self.dataset.DATA_INFOS[split][i]['img']
            image_name = os.path.basename(image_path)
            shutil.copy(image_path, store_path)
            moved_image_path = os.path.join(store_path, image_name)
            img_score_match_list.append(
                (no, moved_image_path, aesthetic_scores[i], classifier_scores[i])
            )
        
        np.savez(os.path.join(store_path, 'img_score_match_list'), img_score_match_list)
            
        print('memory bank has been refreshed')
            
    def mb_load(self, data_path : str, category : str, gt_label : int):
        des_path = os.path.join(data_path, category)
        load_path = os.path.join(self.mb_path, category)
        _lst = np.load(os.path.join(load_path, 'img_score_match_list.npz'), allow_pickle=True)
        arr_0 = _lst['arr_0']
        img_score_match_list = []
        for i in arr_0:
            img_score_match_list.append(i)
        img_score_match_list = sorted(img_score_match_list, key=lambda x: x[2]+x[3], reverse=True)
        self.dataset.DATA_INFOS['memory_bank_category'] = []
        for idx in range(min(self.mb_load_num, len(img_score_match_list))):
            no = img_score_match_list[idx][0]
            image_path = img_score_match_list[idx][1]
            image_name = os.path.basename(image_path)
            shutil.copy(image_path, des_path)
            moved_image_path = os.path.join(des_path, image_name)
            aesthetic_score = img_score_match_list[idx][2]
            self.dataset.DATA_INFOS['memory_bank_category'].append(
                {'no': int(no), 'img': moved_image_path, 'gt_label':gt_label, 'aesthetic_score':float(aesthetic_score)}
            )
        print("memory bank load successfully!")
        fp=open("memory_bank_category.txt", "w")
        print(self.dataset.DATA_INFOS['memory_bank_category'], file=fp)
        fp.close()
        
        
    def merge_mb(self, split='train'):
        if 'memory_bank_category' in self.dataset.DATA_INFOS.keys():
            self.dataset.DATA_INFOS[split] += self.dataset.DATA_INFOS['memory_bank_category']
        else:
            print('First Time Run, No Memory Bank Yet')
