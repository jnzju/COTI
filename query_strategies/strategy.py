import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from getpass import getuser
from socket import gethostname
from utils.progressbar import track_iter_progress
from utils.text import TextLogger
from utils.timer import Timer
from datasets.dataloader import GetDataLoader
from datasets.base_dataset import BaseDataset
from datasets.generated_dataset import generate_virtual_dataset
from evaluation import *
from .utils import get_initialized_cls_module, get_initialized_score_module, get_lr
from architectures.clip import get_clip_image_features
from embedding.embedding import create_embedding, embedding_training
from embedding.hypernetwork import create_hypernetwork, hypernetwork_training
from embedding.preprocess import image_preprocess
import math
import copy
import csv
from torchmetrics.image.inception import InceptionScore
from piq import inception_score
from pytorch_fid.fid_score import calculate_fid_given_paths
import pathlib
from torchvision import transforms
from PIL import ImageFile, Image
import shutil
import clip
import time
import subprocess
import signal
from clip_score.clip_score import calculate_clip_score, DummyDataset
from sklearn.metrics.pairwise import cosine_similarity
from .eval import imagenet_class
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Strategy:
    def __init__(self, dataset: BaseDataset, args, logger):
        self.dataset = dataset
        self.args = args
        
        # models
        self.init_cls_model()
        
        # This is for resume
        self.task_idx = 0
        self.task_idx_path = ''
        self.cls_epoch = 0
        self.scoring_epoch = 0
        self.logger = logger
        self.TextLogger = TextLogger(self.cls_net, vars(args), logger)
        self.timer = Timer()
        
        self.aesthetic_score_list = []
        self.tag_matching_score_list = []
        self.total_score_list = []
        self.fid_score_list = []
        self.is_score_list = []
        self.r_precision_list = []
        self.clip_score_list = []
        
        self.num_labels_list = []
        self.TextLogger._dump_log(vars(args))
        
        score_list = self.predict(self.scoring_net, split='train_full', metric='aesthetic_score')
        for i in range(len(self.dataset.DATA_INFOS['train_full'])):
            self.dataset.DATA_INFOS['train_full'][i]['aesthetic_score'] = score_list[i].item()
        
        # fp=open(os.path.join(self.args.task_path, 'logs.txt'),'w')
        # print(self.dataset.DATA_INFOS['train_full'], file=fp)
        # fp.close()
        
        # use memory bank
        self.mb_setup()
        
        self.dataset.split_tasks()
        
########################### init model   #################################
    def init_cls_model(self) -> None:
        
        train_size = (
            (len(self.dataset.DATA_INFOS['train_full']) + len(self.dataset.DATA_INFOS['train_init'])) / self.args.task_num
        ) * 2
        
        self.cls_net, self.cls_optimizer, self.cls_scheduler = \
            get_initialized_cls_module(lr=self.args.cls_lr, 
                                       momentum=self.args.cls_momentum,
                                       weight_decay=self.args.cls_weight_decay,
                                       optim_type=self.args.cls_optim_type,
                                       T_max=self.args.cls_n_epoch * math.ceil(train_size / self.args.cls_batch_size),
                                       num_classes=2)
        self.scoring_net, self.scoring_optimizer = \
            get_initialized_score_module(self.args.scoring_lr, 
                                         self.args.scoring_momentum,
                                         self.args.scoring_weight_decay)

########################### cls training #################################
    def _classifier_train(self, loader_tr, clf_group: dict, clf_name='train_cls') -> float:
        """Represents one epoch.

        :param loader_tr: (:obj:`torch.utils.data.DataLoader`) The training data wrapped in DataLoader.

        Accuracy and loss in the each iter will be recorded.

        """
        iter_out = self.args.out_iter_freq
        loss_list = []
        right_count_list = []
        samples_per_batch = []
        clf_group['clf'].train()

        iter_num = 0
        total_loss = 0.0
        for batch_idx, (x, y, _, _, _) in enumerate(loader_tr):
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
            
            # dic_acc  = 1.0 * np.sum(right_count_list[-iter_out:]) / np.sum(samples_per_batch[-iter_out:])
            total_loss += np.sum(loss_list[-iter_out:])
            iter_num += iter_out
            # if log_show:
            #     
            
                
        clf_group['scheduler'].step()
        return total_loss / float(iter_num)

    def _scorer_train(self, loader_tr, clf_group: dict, clf_name='train_scoring') -> float:
        """Represents one epoch.

                :param loader_tr: (:obj:`torch.utils.data.DataLoader`) The training data wrapped in DataLoader.

                Accuracy and loss in the each iter will be recorded.

                """
        iter_out = self.args.out_iter_freq
        loss_list = []
        samples_per_batch = []
        clf_group['scoring'].train()
        
        iter_num = 0
        total_loss = 0.0
        for batch_idx, (x, _, _, _, score_y) in enumerate(loader_tr):
            x, score_y = x.cuda(), score_y.to(torch.float).cuda()
            clf_group['optimizer'].zero_grad()
            out = torch.flatten(clf_group['scoring'](x))
            samples_per_batch.append(len(score_y))
            loss = F.mse_loss(out, score_y)

            loss_list.append(loss.item())
            loss.backward()
            clf_group['optimizer'].step()
            iter_time = self.timer.since_last_check()

            total_loss += np.sum(loss_list[-iter_out:])
            iter_num += iter_out
        
        return total_loss / float(iter_num)

    def cls_train(self):
        self.logger.info('Start running, host: %s, work_dir: %s',
                         f'{getuser()}@{gethostname()}', self.args.work_dir)
        self.logger.info('max: %d classification epochs', self.args.cls_n_epoch)
        self.logger.info('max: %d scoring epochs', self.args.scoring_n_epoch)
        self.cls_net.train()
        self.scoring_net.train()

        loader_tr_cls = GetDataLoader(self.dataset, split='train', shuffle=True, batch_size=self.args.cls_batch_size,
                                      num_workers=self.args.cls_num_workers, task='cls')
        loader_tr_scoring = GetDataLoader(self.dataset, split='task_now', shuffle=True,
                                          batch_size=self.args.cls_batch_size,
                                          num_workers=self.args.scoring_num_workers, task='scoring')
        
        cls_loss = 0
        while self.cls_epoch < self.args.cls_n_epoch:
            self.timer.since_last_check()
            new_cls_loss = self._classifier_train(loader_tr_cls,
                                   {'clf': self.cls_net, 'optimizer': self.cls_optimizer, 'scheduler': self.cls_scheduler})
            self.logger.info(f'mode: tag matching, loss: {new_cls_loss}')
            self.cls_epoch += 1
            if abs(new_cls_loss - cls_loss) < 0.1:
                break
            cls_loss = new_cls_loss
            
        score_loss = 0
        while self.scoring_epoch < self.args.scoring_n_epoch:
            self.timer.since_last_check()
            new_score_loss = self._scorer_train(loader_tr_scoring,
                               {'scoring': self.scoring_net, 'optimizer': self.scoring_optimizer})
            self.logger.info(f'mode: scoring, loss: {new_score_loss}')
            self.scoring_epoch += 1
            
            if (new_score_loss < 2.5) and (abs(new_score_loss - score_loss) < 0.3):
                break
            score_loss = new_score_loss

        self.cls_epoch = 0
        self.scoring_epoch = 0
        # self.save()
        self.cls_net.eval()
        self.scoring_net.eval()

########################### predict score ######################################
    def predict(self, clf, split, metric):
        if isinstance(clf, torch.nn.Module):
            clf.eval()

        # Predicting classification model quality
        loader = GetDataLoader(self.dataset, split=split,
                                shuffle=False,
                                batch_size=self.args.validation_batch_size, task='scoring')
        print('\n')
        print(f"Calculating Informativeness with {metric} on {split}...")
        print('\n')
        # 求图像得分的平均值
        # calculating scores
        if metric == 'aesthetic_score':
            result = []
            with torch.no_grad():
                for x, _, _, _, _ in track_iter_progress(loader, "predict aesthetic score"):
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
                pred = torch.zeros([len(self.dataset.DATA_INFOS[split]), 2]).cuda()
                
                with torch.no_grad():
                    for x, _, _, idxs, _ in track_iter_progress(loader, "predict tag matching score"):
                        x = torch.squeeze(x, 1).cuda()
                        out = clf(x)
                        pred[idxs] += F.softmax(out, dim=1)
                result = torch.flatten(pred[:, :1] * 10.0)
            elif self.args.tag_matching_strategy == 'representation_distance':
                pred = torch.zeros([len(self.dataset.DATA_INFOS[split]), 2]).cuda()
                with torch.no_grad():
                    for _, _, _, idxs, _ in track_iter_progress(loader, "predict tag matching score"):
                        num = len(idxs)
                        image_paths = [self.dataset.DATA_INFOS[split][idxs[i]]['img'] for i in range(num)]
                        positive_text_input = "a " + self.args.category
                        sub_text_input = "a " + self.args.subcategory
                        device = "cuda"
                        model, preprocess = clip.load("ViT-B/32", device=device)
                        
                        for i in range(num):
                            image = preprocess(Image.open(image_paths[i])).unsqueeze(0).to(device)
                            text = clip.tokenize([positive_text_input, sub_text_input]).to(device)
                            logits_per_image, logits_per_text = model(image, text)
                            pred[idxs[i]] = logits_per_image.softmax(dim=-1)
                result = torch.flatten(pred[:, 0] * 10.0)
            else :
                raise NotImplementedError
        elif metric == 'r_precision':
            # raise NotImplementedError
            # return 1
            gen_data_dir = os.path.join(self.task_idx_path, 'gen_val_images')
            device = "cuda"
            model, preprocess = clip.load("ViT-B/32", device=device)
            
            template_list = []
            template_list.append(f'a painting of {self.args.category}')
            for i in range(0,1000):
                template_list.append("a painting of " + imagenet_class[i])
            text = clip.tokenize(template_list).to(device)
            
            file_paths = list(pathlib.Path(gen_data_dir).glob('*.png'))
            image = torch.stack(
                ([preprocess(Image.open(img_path)).to(device) for img_path in file_paths])
            )
            
            with torch.no_grad():
                text = model.encode_text(text)
                image = model.encode_image(image)
                image = image.float()
                similarity_matrix = torch.from_numpy(cosine_similarity(image.cpu(), text.cpu()))
                
                labels = torch.ones(len(os.listdir(gen_data_dir))) * 0
                labels = labels.view(-1, 1)
                _, indices = torch.sort(similarity_matrix, descending=True)
                top = indices[:,:200]
                num = 0
                for i in range(len(labels)):
                    if 0 in top[i]:
                        num += 1
                r_precision = num /top.shape[0]
            
            return r_precision

        elif metric == 'is':
            transform=transforms.Compose([
                transforms.Resize(size=299),
                transforms.RandomCrop(size=299),
                transforms.ToTensor()
            ])
            file_paths = list(pathlib.Path(os.path.join(self.task_idx_path, 'gen_val_images')).glob('*.png'))
            gen_tensor = torch.stack(
                ([transform(Image.open(img_path)).to(torch.uint8) for img_path in file_paths])
            )
            inception = InceptionScore()
            inception.update(gen_tensor)
            score = inception.compute()
            return score[0].item()

            # another implementation
            # device = 'cuda'
            # model, preprocess = clip.load("ViT-B/32", device=device)
            
            # file_paths = list(pathlib.Path(os.path.join(self.task_idx_path, 'gen_val_images')).glob('*.png'))
            # gen_tensor = torch.stack(
            #     ([preprocess(Image.open(img_path)).to('cuda') for img_path in file_paths])
            # )
            
            # with torch.no_grad():
            #     out = model.encode_image(gen_tensor)
            
            # out = out.cpu().float()
            # score = inception_score(out)
            # return score

            
        elif metric == 'fid':
            init_path = os.path.join(self.args.training_dataset_initial, self.args.category)
            gen_path = os.path.join(self.task_idx_path, 'gen_val_images')
            paths = [init_path, gen_path]
            return calculate_fid_given_paths(paths, batch_size=len(os.listdir(init_path)), device='cuda', dims=2048, num_workers=0, transforms=
                                      transforms.Compose([
                                        transforms.Resize(size=299),
                                        transforms.RandomCrop(size=299),
                                        transforms.ToTensor(),
                                        ]))
        elif metric == 'clip':
            
            os.makedirs(os.path.join(self.task_idx_path,  'gen_texts'), mode=0o777)
            fake_path = os.path.join(self.task_idx_path, 'gen_texts')
            real_path = os.path.join(self.task_idx_path, 'gen_val_images')
            num_images = len(os.listdir(real_path))
            for i in range(num_images):
                try:
                    with open(os.path.join(fake_path, f'output_{i}.txt'), 'w') as file:
                        file.write(f'a painting of {self.args.category}')
                except Exception as e:
                    print(f"文件创建失败：{e}")

            print('Loading CLIP model: ViT-B/32')
            model, preprocess = clip.load('ViT-B/32', device='cuda')
            
            dataset = DummyDataset(real_path, fake_path,
                                'img', 'txt',
                                transform=preprocess, tokenizer=clip.tokenize)
            dataloader = DataLoader(dataset, 50, 
                                    num_workers=0, pin_memory=True)
            
            print('Calculating CLIP Score:')
            clip_score = calculate_clip_score(dataloader, model, 'img', 'txt')
            clip_score = clip_score.cpu().item()
            
            return clip_score
        else:
            raise NotImplementedError
        # n_drops ignored
        return result
        # back to train split as the default split

########################### embedding training #################################
    def _embedding_train(self, lr, temp_preprocessed_path, begin, end, iter_step, save_embedding_every):
        best_score = 0
        begin = begin
        end = end
        embedding_iters = begin
        while embedding_iters < end:
            embedding_path_list, image_path_list = \
                embedding_training(self.args.server,
                                    self.args.stable_diffusion_model_path,
                                    self.args.stable_diffusion_url,
                                    self.args.category,
                                    learn_rate=lr,
                                    data_root=temp_preprocessed_path,
                                    log_directory=os.path.join(self.task_idx_path, "embedding"),
                                    steps=embedding_iters + iter_step,
                                    initial_step=embedding_iters,
                                    save_embedding_every=save_embedding_every,
                                    template_filename="style_filewords.txt",
                                    preview_prompt=f"a_painting_of_{self.args.category}, "
                                                    f"{self.args.category}, real_life")
            
            # 接下来筛选最优embedding
            self.dataset.DATA_INFOS['temp'] = [{'img': path, 'gt_label': 0, 'aesthetic_score': 0., 'type': 'temp'} for _, path in enumerate(image_path_list)]
            aesthetic_score_list = self.predict(self.scoring_net, split='temp', metric='aesthetic_score')
            tag_matching_score_list = self.predict(self.cls_net, split='temp', metric='tag_matching_score')
            total_score_list = aesthetic_score_list + tag_matching_score_list
            best_idx = torch.argmax(total_score_list).item()
            
            if total_score_list[best_idx] > best_score:
                best_score = total_score_list[best_idx]
                embedding_pt_dict = torch.load(embedding_path_list[best_idx])
                embedding_pt_dict['name'] = self.args.category
                dsc_file = os.path.join(self.args.stable_diffusion_model_path, "embeddings", f"{self.args.category}.pt")
                os.remove(dsc_file)
                torch.save(embedding_pt_dict, dsc_file)
            # self.logger.info(f"embedding_iters:{embedding_iters}, idx:{best_idx}, embedding path:{embedding_path_list[best_idx]}, image path:{image_path_list[best_idx]}, aesthetic score:{aesthetic_score_list[best_idx]}, tag matching score:{tag_matching_score_list[best_idx]}, total score:{total_score_list[best_idx]}, lr: {lr}")
            else:
                break
            embedding_iters += iter_step
        del self.dataset.DATA_INFOS['temp']
        return embedding_iters-iter_step
        
    def embedding_train(self):
        temp_origin_path = os.path.join(self.task_idx_path, 'origin')
        temp_preprocessed_path = os.path.join(self.task_idx_path, 'prepocess')
        os.makedirs(temp_origin_path, mode=0o777, exist_ok=True)
        os.makedirs(temp_preprocessed_path, mode=0o777, exist_ok=True)
        
        selected_images = self.dataset.DATA_INFOS['task_now']
        for data_dict in selected_images:
            shutil.copy(data_dict['img'], temp_origin_path)
        image_preprocess(self.args.server, self.args.stable_diffusion_url, self.args.category, temp_origin_path, temp_preprocessed_path,
                         768, 768, "ignore", False, False, False)
        
        
        create_embedding(self.args.stable_diffusion_url, self.args.stable_diffusion_model_path, self.args.category, overwrite_old=True)
        os.makedirs(os.path.join(self.task_idx_path, "embedding"), mode=0o777, exist_ok=True)
        
        # 四个lr不同种训练
        
        #第一轮
        lrs = self.args.embedding_learn_rate
        
        new_begin = self._embedding_train(
            lr=lrs[0],
            temp_preprocessed_path=temp_preprocessed_path,
            begin=0,
            end=10000,
            iter_step=1000,
            save_embedding_every=100
        )
        
        new_begin = self._embedding_train(
            lr=lrs[1],
            temp_preprocessed_path=temp_preprocessed_path,
            begin=new_begin,
            end=new_begin+1000,
            iter_step=100,
            save_embedding_every=10
        )
            
        new_begin = self._embedding_train(
            lr=lrs[2],
            temp_preprocessed_path=temp_preprocessed_path,
            begin=new_begin,
            end=new_begin+100,
            iter_step=10,
            save_embedding_every=1
        )   
            
        new_begin = self._embedding_train(
            lr=lrs[3],
            temp_preprocessed_path=temp_preprocessed_path,
            begin=new_begin,
            end=new_begin+10,
            iter_step=1,
            save_embedding_every=1
        )

########################### hypernetwork trainning #############################
    def hypernetwork_train(self):
        temp_origin_path = os.path.join(self.task_idx_path, 'origin')
        temp_preprocessed_path = os.path.join(self.task_idx_path, 'prepocess')
        os.makedirs(temp_origin_path, mode=0o777, exist_ok=True)
        os.makedirs(temp_preprocessed_path, mode=0o777, exist_ok=True)
        
        selected_images = self.dataset.DATA_INFOS['task_now']
        for data_dict in selected_images:
            shutil.copy(data_dict['img'], temp_origin_path)
        image_preprocess(self.args.server, self.args.stable_diffusion_url, self.args.category, temp_origin_path, temp_preprocessed_path,
                         768, 768, "ignore", False, False, False)
        
        hypernetwork_iters = 0
        create_hypernetwork(self.args.stable_diffusion_url, self.args.stable_diffusion_model_path, self.args.category, overwrite_old=True)
        os.makedirs(os.path.join(self.task_idx_path, "hypernetwork"), mode=0o777, exist_ok=True)
        
        first_flag=True
        for lr in self.args.hypernetwork_learn_rate:
            self.logger.info('---------------------------------------------------------')
            self.logger.info(f"Training with learning rate {lr} at task {self.task_idx}!")
            hypernetwork_path_list, image_path_list = \
                hypernetwork_training(
                                      self.args.stable_diffusion_model_path,
                                      self.args.stable_diffusion_url,
                                      self.args.category,
                                      learn_rate=lr,
                                      data_root=temp_preprocessed_path,
                                      log_directory=os.path.join(self.task_idx_path, "hypernetwork"),
                                      steps=hypernetwork_iters + self.args.hypernetwork_steps_per_lr,
                                      initial_step=hypernetwork_iters,
                                      save_hypernetwork_every=self.args.save_hypernetwork_every,
                                      template_filename="hypernetwork.txt",
                                      preview_prompt=f'a_photo_of_<hyper:{self.args.category}:1.0>, ' + self.args.category + ", real_life")

            # 接下来筛选最优hypernetwork
            self.dataset.DATA_INFOS['temp'] = [{'img': path, 'gt_label': 0, 'aesthetic_score': 0., 'type': 'temp'
                                                } for _, path in enumerate(image_path_list)]
            aesthetic_score_list = self.predict(self.scoring_net, split='temp', metric='aesthetic_score')
            tag_matching_score_list = self.predict(self.cls_net, split='temp', metric='tag_matching_score')
            hypernetwork_iters = hypernetwork_iters + self.args.hypernetwork_steps_per_lr
            total_score_list = aesthetic_score_list + tag_matching_score_list
            best_idx = torch.argmax(total_score_list).item()
            if first_flag:
                best_score = total_score_list[best_idx]
                best_lr = lr
                first_flag=False
            elif total_score_list[best_idx] >= best_score:
                self.logger.info(f"Hypernetwork weight is being replaced! Former score:{best_score} lr:{best_lr} Now score:{total_score_list[best_idx]} lr:{lr}")
                
                best_score=total_score_list[best_idx]
                best_lr = lr
            
                # 选择完毕，移动最佳embedding替换，删除多余的embedding
                hypernetwork_pt_dict = torch.load(hypernetwork_path_list[best_idx])
                hypernetwork_pt_dict['name'] = self.args.category
                dsc_file = os.path.join(self.args.stable_diffusion_model_path, "models", "hypernetworks", f"{self.args.category}.pt")
                os.remove(dsc_file)
                # shutil.copy(hypernetwork_path_list[best_idx], dsc_file)
                torch.save(hypernetwork_pt_dict, dsc_file)
                """
                for del_idx in range(best_idx + 1, self.args.hypernetwork_steps_per_lr // self.args.save_hypernetwork_every):
                    os.remove(image_path_list[del_idx])
                    os.remove(hypernetwork_path_list[del_idx])
                """
            del self.dataset.DATA_INFOS['temp']
  
########################### metric and review ##################################
    def gen_validation_set(self):
        print(f'Generating validation images: category: {self.args.category}, num: {self.args.validation_generated_images_per_class}')
        
        if self.args.sd_train_method == 'embed':
            prompt = f"a_painting_of_{self.args.category}, " + self.args.category + ", real_life"
        elif self.args.sd_train_method == 'hyper':
            prompt = f'a_photo_of_<hyper:{self.args.category}:1.0>, ' + self.args.category + ", real_life"
        
        self.dataset.DATA_INFOS['val'+f'{self.task_idx}'] = generate_virtual_dataset(
            url=self.args.stable_diffusion_url,
            prompt=prompt,
            num_samples=self.args.validation_generated_images_per_class,
            temp_dir=os.path.join(self.task_idx_path, "gen_val_images"))

    def get_metrics(self):
        split = 'val' + f'{self.task_idx}'
        
        aesthetic_score_all = self.predict(self.scoring_net, split=split, metric='aesthetic_score')
        aesthetic_score_all[aesthetic_score_all > 10.0] = 10.0
        self.aesthetic_score_list.append(torch.mean(aesthetic_score_all).item())
        
        tag_matching_score_all = self.predict(self.cls_net, split=split, metric='tag_matching_score')
        self.tag_matching_score_list.append(torch.mean(tag_matching_score_all).item())
        
        self.total_score_list.append(10 * math.sin(self.aesthetic_score_list[-1] * np.pi / 20)
                                        * math.sin(self.tag_matching_score_list[-1] * np.pi / 20))

        r_precision = self.predict(None, split=split, metric='r_precision')
        self.r_precision_list.append(r_precision)
        
        self.is_score_list.append(self.predict(None, split=split, metric='is'))
        
        self.fid_score_list.append(self.predict(None, split=split, metric='fid'))
        
        self.clip_score_list.append(self.predict(None, split=split, metric='clip'))

        log_dict = dict(mode='val', task=self.task_idx,
                        aesthetic=self.aesthetic_score_list[-1],
                        tag_matching=self.tag_matching_score_list[-1],
                        total=self.total_score_list[-1],
                        r_precision=self.r_precision_list[-1],
                        is_score=self.is_score_list[-1],
                        fid_score=self.fid_score_list[-1],
                        clip_score=self.clip_score_list[-1])
        self.TextLogger.log(log_dict)

    def record_evaluation_results(self):
        file_name = os.path.join(self.args.work_dir, 'evaluation.csv')
        header = ['task_idx', 'aesthetic', 'tag_matching', 'total', 'is', 'fid', 'r_precision', 'clip']
        with open(file_name, 'w', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(header)
            
            for i in range(len(self.aesthetic_score_list)):
                f_csv.writerow([i,
                                self.aesthetic_score_list[i],
                                self.tag_matching_score_list[i],
                                self.total_score_list[i],
                                self.is_score_list[i],
                                self.fid_score_list[i],
                                self.r_precision_list[i],
                                self.clip_score_list[i]])


########################### run ##############################################
    def run(self):
        
        while self.task_idx < self.args.task_num:
            if self.args.sd_train_method == 'embed':
                process = subprocess.Popen('CUDA_VISIBLE_DEVICES=1 ./webui.sh --port 7860', shell=True, cwd=self.args.stable_diffusion_model_path)
                time.sleep(20) # model loading
            else:
                raise NotImplementedError
            
            self.task_idx_path = os.path.join(self.args.work_dir, f"task_{self.task_idx}")
            os.makedirs(self.task_idx_path, mode=0o777, exist_ok=True)
            
            # prepare data
            # 切换到下一个任务
            self.dataset.task_idx += 1
            self.dataset.DATA_INFOS['train_now'] = copy.deepcopy(self.dataset.DATA_INFOS['tasks'][self.dataset.task_idx])
            split_length = len(self.dataset.DATA_INFOS['train_now'])
            idxs = np.arange(split_length)
            aesthetic_scores=self.predict(self.scoring_net, split='train_now', metric='aesthetic_score')
            tag_matching_scores=self.predict(self.cls_net, split='train_now', metric='tag_matching_score')
            total_scores = (aesthetic_scores + tag_matching_scores).sort()[1].cpu().numpy() #[1]表示的是使用返回值的索引
            idxs_touse = idxs[total_scores]
            num_touse = len(self.dataset.DATA_INFOS['train_now']) // 2
            lst = []
            cnt = 0
            for i in idxs_touse:
                if cnt > num_touse:
                    break
                lst.append(self.dataset.DATA_INFOS['train_now'][i])
                cnt += 1
            self.dataset.DATA_INFOS['train_now'] = lst
            
            
            # 将mb中的优质数据load出来训练
            self.mb_load()                
            
            # 训练集中加入分类负样本
            self.dataset.DATA_INFOS['train'] = copy.deepcopy(self.dataset.DATA_INFOS['task_now']) + list(np.random.choice(self.dataset.DATA_INFOS['train_sub'],
                                                                                                        min(len(self.dataset.DATA_INFOS['task_now']),
                                                                                                            len(self.dataset.DATA_INFOS['train_sub'])),
                                                                                                        replace=False))            
            
            self.logger.info('\n')
            self.logger.info(f'Active Continual Learning, training task_{self.dataset.task_idx} with {len(self.dataset.DATA_INFOS["task_now"])} samples')
            self.logger.info('\n')
            
            if self.args.sd_train_method == 'embed':
                self.embedding_train()              # 根据训练集训练embedding
                process.send_signal(signal.SIGINT)
                time.sleep(20)
            else:
                raise NotImplementedError
            
            self.cls_train()                    # 根据训练集训练分类器和评分器
            
            self.mb_store()                     # 对训练集中的正样本得分较高的数据存储到mb中
            
            if self.args.sd_train_method == 'embed':
                process = subprocess.Popen('CUDA_VISIBLE_DEVICES=1 ./webui.sh --port 7860', shell=True, cwd=self.args.stable_diffusion_model_path)
                time.sleep(20) # model loading
            else:
                raise NotImplementedError
            
            

            self.gen_validation_set()           # 生成样本
            
            self.get_metrics()                  # 对样本的质量进行评估，得出embedding质量是否提高

            self.task_idx += 1
            
            if self.args.sd_train_method == 'embed':
                process.send_signal(signal.SIGINT)
                time.sleep(20)
            else:
                raise NotImplementedError

        self.record_evaluation_results()
        
#########################  memory bank mechanism realization  ########################
    def load_img_score_match_list(self, load_path) -> list :
        img_score_match_list = []
        if not os.path.exists(os.path.join(load_path, 'img_score_match_list.npz')):
            print('No img_score_match_list.npz. Make sure this is the first task.')
            return img_score_match_list
        _lst = np.load(os.path.join(load_path, 'img_score_match_list.npz'), allow_pickle=True)
        arr_0 = _lst['arr_0']
        
        for i in arr_0:
            img_score_match_list.append(
                (i[0], float(i[1]), float(i[2]))
            )
        return img_score_match_list
    
    def mb_setup(self): #data_path是指到training_dataset
        if not os.path.exists(self.args.mb_path):
            print('[Memory Bank Path ERROR] Please Specify Your Memory Bank Path!')
            exit(1)
        
        if os.path.exists(os.path.join(self.args.mb_path, self.args.category, 'img_score_match_list.npz')):
            print('Because of experiment\'s need, now memorybank/{category}\'s data should be cleared.')
            shutil.rmtree(os.path.join(self.args.mb_path, self.args.category))
            os.makedirs(os.path.join(self.args.mb_path, self.args.category), mode=0o777, exist_ok=True)
        elif os.path.exists(os.path.join(self.args.mb_path, self.args.category)):
            print(f'memorybank/{self.args.category} being used is damaged, COTI needs to regenerate it and remove previous image data.')
            shutil.rmtree(os.path.join(self.args.mb_path, self.args.category))
            os.makedirs(os.path.join(self.args.mb_path, self.args.category), mode=0o777, exist_ok=True)
        else:
            os.makedirs(os.path.join(self.args.mb_path, self.args.category), mode=0o777, exist_ok=True)
    
    def mb_store(self, split='task_now'): # select top mb_selected_num images and save them to the memory bank
        split_length = len(self.dataset.DATA_INFOS[split])
        idxs = np.arange(split_length)
        aesthetic_scores=self.predict(self.scoring_net, split=split, metric='aesthetic_score')
        tag_matching_scores=self.predict(self.cls_net, split=split, metric='tag_matching_score')
        total_scores = (aesthetic_scores + tag_matching_scores).sort()[1].cpu().numpy() #[1]表示的是使用返回值的索引
        idxs_tostore = idxs[total_scores]
        
        aesthetic_scores = aesthetic_scores.cpu().numpy()
        tag_matching_scores = tag_matching_scores.cpu().numpy()
        
        img_score_match_list = []
        
        img_score_match_list = self.load_img_score_match_list(os.path.join(self.args.mb_path, self.args.category))

        select_num = 0
        for i in idxs_tostore:
            if select_num > self.args.mb_save_num:
                break
            if self.dataset.DATA_INFOS[split][i]['type'] != 'mb':
                image_path = self.dataset.DATA_INFOS[split][i]['img']
                image_name = os.path.basename(image_path)
                shutil.copy(image_path, os.path.join(self.args.mb_path, self.args.category))
                moved_image_path = os.path.join(self.args.mb_path, self.args.category, image_name)
                img_score_match_list.append(
                    (moved_image_path, 10.0 - 2.0 * float(select_num) / float(self.args.mb_save_num), tag_matching_scores[i])
                )
                select_num += 1
            
        img_score_match_list = sorted(img_score_match_list, key=lambda x: x[1]+x[2], reverse=True)
        
        if len(img_score_match_list) > self.args.mb_size:
            for i in range(self.args.mb_size, len(img_score_match_list)):
                os.remove(img_score_match_list[i][0])
            img_score_match_list = img_score_match_list[:self.args.mb_size]
        
        np.savez(os.path.join(self.args.mb_path, self.args.category, 'img_score_match_list'), img_score_match_list)
            
        print('memory bank has been refreshed')
            
    def mb_load(self, tosplit='task_now'): # 把memory bank中的部分数据load到train中
        load_path = os.path.join(self.args.mb_path, self.args.category)
        if os.path.exists(os.path.join(load_path, 'img_score_match_list.npz')):
            
            img_score_match_list = self.load_img_score_match_list(load_path)
            for idx in range(min(self.args.mb_load_num, len(img_score_match_list))):
                # (moved_image_path, aesthetic_scores[i], tag_matching_scores[i])
                image_path = img_score_match_list[idx][0]
                aesthetic_score = img_score_match_list[idx][1]
                self.dataset.DATA_INFOS[tosplit].append(
                    {'img': image_path, 'gt_label':0, 'aesthetic_score':float(aesthetic_score), 'type': 'mb'}
                )
            print("memory bank load successfully!")
        else:
            print("memory bank is empty yet.")