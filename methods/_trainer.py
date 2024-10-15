import os
import gc
import argparse
import datetime
import logging
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from PIL import Image
from randaugment import RandAugment
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import get_dataset
from models import get_model
from utils.augment import Cutout
from utils.indexed_dataset import IndexedDataset
from utils.memory import Memory
from utils.online_sampler import OnlineSampler, OnlineTestSampler
from utils.train_utils import select_optimizer, select_scheduler
from torch.utils.data import DataLoader, DistributedSampler


##################################################################
# This is trainer with a DistributedDataParallel                 #
# Based on the following tutorial:                               #
# https://github.com/pytorch/examples/blob/main/imagenet/main.py #
# And Deit by FaceBook                                           #
# https://github.com/facebookresearch/deit                       #
##################################################################
class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(self.pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label

    def pil_loader(self, path):
        """
        Ref:
        https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
        """
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

class _Trainer():

    def __init__(self, *args, **kwargs) -> None:

        self.args = kwargs

        self.method = kwargs.get("method")

        self.n = kwargs.get("n")
        self.m = kwargs.get("m")
        self.rnd_NM = kwargs.get("rnd_NM")

        self.n_tasks = kwargs.get("n_tasks")
        self.epochNum = kwargs.get("epochNum")
        self.dataset_name = kwargs.get("dataset")
        self.rnd_seed = kwargs.get("rnd_seed")

        self.memory_size = kwargs.get("memory_size")
        self.log_path = kwargs.get("log_path")
        self.model_name = kwargs.get("model_name")
        self.opt_name = kwargs.get("opt_name")
        self.sched_name = kwargs.get("sched_name")
        self.batchsize = kwargs.get("batchsize")
        self.n_worker = kwargs.get("n_worker")
        self.lr = kwargs.get("lr")
        self.init_model = kwargs.get("init_model")
        self.init_opt = kwargs.get("init_opt")
        self.topk = kwargs.get("topk")
        self.use_amp = kwargs.get("use_amp")
        self.transforms = kwargs.get("transforms")
        self.reg_coef = kwargs.get("reg_coef")
        self.data_dir = kwargs.get("data_dir")
        self.debug = kwargs.get("debug")
        self.note = kwargs.get("note")
        self.selection_size = kwargs.get("selection_size")
        self.ca = kwargs.get("ca")
        self.ssca = kwargs.get("ssca")
        self.ca_epochs = kwargs.get("ca_epochs")
        self.model_type = kwargs.get("model_type")
        self.feature_dim = kwargs.get("feature_dim")
        self.num_prompt = kwargs.get("num_prompt")
        self.n_ctx = kwargs.get("n_ctx")
        self.topK = kwargs.get("topK")
        self.text_template = kwargs.get("text_template")
        self.wd = 0.0
        self.task_id = 0
        self.disjoint_classes = None
        self.disjoint_class_names = None
        self.disjoint_class_num = None

        self.eval_period = kwargs.get("eval_period")
        self.temp_batchsize = kwargs.get("temp_batchsize")
        self.online_iter = kwargs.get("online_iter")
        self.num_gpus = kwargs.get("num_gpus")
        self.workers_per_gpu = kwargs.get("workers_per_gpu")
        self.imp_update_period = kwargs.get("imp_update_period")


        self.zero_shot_evaluation = kwargs.get("zero_shot_evaluation")
        self.zero_shot_dataset = kwargs.get("zero_shot_dataset")

        # for distributed training
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'

        self.lr_step = kwargs.get("lr_step")  # for adaptive LR
        self.lr_length = kwargs.get("lr_length")  # for adaptive LR
        self.lr_period = kwargs.get("lr_period")  # for adaptive LR

        self.memory_epoch = kwargs.get("memory_epoch")  # for RM
        self.distilling = kwargs.get("distilling")  # for BiC
        self.agem_batch = kwargs.get("agem_batch")  # for A-GEM
        self.mir_cands = kwargs.get("mir_cands")  # for MIR

        self.start_time = time.time()
        self.num_updates = 0
        self.train_count = 0
        self._known_classes = 0
        self._total_classes = 0

        self.dtype = None


        self.ngpus_per_nodes = torch.cuda.device_count()
        # self.ngpus_per_nodes = 3
        self.world_size = 1
        if "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"] != '':
            self.world_size = int(
                os.environ["WORLD_SIZE"]) * self.ngpus_per_nodes
        else:
            self.world_size = self.world_size * self.ngpus_per_nodes#4
        self.distributed = self.world_size > 1

        if self.distributed:
            self.batchsize = self.batchsize // self.world_size#64,4,16
        if self.temp_batchsize is None:
            self.temp_batchsize = self.batchsize // 2
        if self.temp_batchsize > self.batchsize:
            self.temp_batchsize = self.batchsize
        # self.memory_batchsize = self.batchsize - self.temp_batchsize
        self.memory_batchsize = 0

        if 'debug' not in self.note:
            self.log_dir = os.path.join(
                self.log_path, self.dataset_name,
                f"TASK{self.n_tasks}N{self.n}M{self.m}",
                f"{self.note}_{datetime.datetime.now().strftime('%y%m%d%H')}")
        else:
            self.log_dir = os.path.join(self.log_path, "debug")
        os.makedirs(self.log_dir, exist_ok=True)

    def setup_distributed_model(self):
        logging.info("Building model...")
        self.model = self.model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.model.to(self.device)
        self.model_without_ddp = self.model
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,device_ids=[self.gpu],output_device=self.gpu)
            self.model._set_static_graph()
            self.model_without_ddp = self.model.module
        self.criterion = self.model_without_ddp.loss_fn if hasattr(
            self.model_without_ddp, "loss_fn") else nn.CrossEntropyLoss(
                reduction="mean")
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

        n_params = sum(p.numel() for p in self.model_without_ddp.parameters())
        logging.info(f"Total Parameters :\t{n_params}")
        n_params = sum(p.numel() for p in self.model_without_ddp.parameters()
                       if p.requires_grad)
        logging.info(f"Learnable Parameters :\t{n_params}")

    def setup_zero_shot_dataset(self, dataset_name):
        dataset, mean, std, _ = get_dataset(dataset_name)
        test_transform = transforms.Compose([
            transforms.Resize((self.inp_size, self.inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_dataset = dataset(root=self.data_dir,
                               train=False,
                               download=True,
                               transform=test_transform)
        classes_names = test_dataset.classes_names

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=self.batchsize,
                                     shuffle=False,
                                     num_workers=self.n_worker,
                                     pin_memory=True)
        return test_dataloader, classes_names

    def setup_dataset(self):
        # get dataset
        self.train_dataset = self.dataset(root=self.data_dir,
                                          train=True,
                                          download=True,
                                          transform=transforms.ToTensor())
        self.test_dataset = self.dataset(root=self.data_dir,
                                         train=False,
                                         download=True,
                                         transform=self.test_transform)
        self.n_classes = len(self.train_dataset.classes)

        self.exposed_classes = []
        self.exposed_classes_names = []
        self.seen = 0


    def get_dataset_by_indices(
            self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self.train_dataset.data, np.array(
            self.train_dataset.targets
        )
        elif source == "test":
            x, y = self.test_dataset.data, np.array(
            self.test_dataset.targets
        )
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = self.train_transform
        elif mode == "test":
            trsf = self.test_transform
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )

            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf)
        else:
            return DummyDataset(data, targets, trsf)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]


    def setup_transforms(self):
        train_transform = []
        self.cutmix = "cutmix" in self.transforms
        if "autoaug" in self.transforms:
            train_transform.append(lambda x: (x * 255).type(torch.uint8))
            if 'cifar' in self.dataset_name:
                train_transform.append(
                    transforms.AutoAugment(
                        transforms.AutoAugmentPolicy('cifar10')))
            elif 'imagenet' in self.dataset_name:
                train_transform.append(
                    transforms.AutoAugment(
                        transforms.AutoAugmentPolicy('imagenet')))
            elif 'svhn' in self.dataset_name:
                train_transform.append(
                    transforms.AutoAugment(
                        transforms.AutoAugmentPolicy('svhn')))
            train_transform.append(lambda x: x.type(torch.float32) / 255)

        if "cutout" in self.transforms:
            train_transform.append(Cutout(size=16))
        if "randaug" in self.transforms:
            train_transform.append(RandAugment())

        self.train_transform = transforms.Compose([
            *train_transform,
            transforms.Resize((self.inp_size, self.inp_size)),
            transforms.RandomCrop(self.inp_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(self.mean, self.std),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((self.inp_size, self.inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def run(self):
        # Distributed Launch
        if self.ngpus_per_nodes > 1:# 4个gpu
            mp.spawn(self.main_worker, nprocs=self.ngpus_per_nodes, join=True)
        else:
            self.main_worker(0)

    def main_worker(self, gpu) -> None:
        self.gpu = gpu % self.ngpus_per_nodes
        self.device = torch.device(self.gpu)
        if self.distributed:#True
            self.local_rank = self.gpu
            if 'SLURM_PROCID' in os.environ.keys():
                self.rank = int(os.environ['SLURM_PROCID']
                                ) * self.ngpus_per_nodes + self.gpu
                print(
                    f"| Init Process group {os.environ['SLURM_PROCID']} : {self.local_rank}"
                )
            elif 'WORLD_SIZE' in os.environ.keys():
                self.rank = int(
                    os.environ['RANK']) * self.ngpus_per_nodes + self.gpu
                print(
                    f"| Init Process group {os.environ['RANK']} : {self.local_rank}"
                )
            else:
                self.rank = self.gpu# 1
                print(f"| Init Process group 0 : {self.local_rank}")
            if 'MASTER_ADDR' not in os.environ.keys():
                os.environ['MASTER_ADDR'] = '127.0.0.1'#2
                os.environ['MASTER_PORT'] = '12701'
            torch.cuda.set_device(self.gpu)
            time.sleep(self.rank * 0.1)  # prevent port collision
            dist.init_process_group(backend=self.dist_backend,
                                    init_method=self.dist_url,
                                    world_size=self.world_size,
                                    rank=self.rank)
            torch.distributed.barrier()
            self.setup_for_distributed(self.is_main_process())
        else:
            self.setup_for_distributed(True)

        logging.info(str(self.args))

        if self.rnd_seed is not None:
            random.seed(self.rnd_seed)
            np.random.seed(self.rnd_seed)
            torch.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed_all(self.rnd_seed)  # if use multi-GPU
            cudnn.deterministic = True
            logging.info('You have chosen to seed training. '
                         'This will turn on the CUDNN deterministic setting, '
                         'which can slow down your training considerably! '
                         'You may see unexpected behavior when restarting '
                         'from checkpoints.')
        cudnn.benchmark = False

        logging.info(f"Select a CIL method ({self.method})")

        self.dataset, self.mean, self.std, self.n_classes = get_dataset(
            self.dataset_name)
        logging.info(f"Building model ({self.model_name})")
        self.model, self.inp_size = get_model(
            model_name=self.model_name,
            method=self.method,
            num_classes=self.n_classes,
            device=self.device,
            peft_encoder=self.args['peft_encoder'],
            args = argparse.Namespace(**self.args)
        )
        # self.dtype = self.model.model.dtype

        self.setup_transforms()
        self.setup_dataset()
        self.setup_distributed_model()
        self.memory = Memory()
        self.total_samples = len(self.train_dataset)

        train_dataset = IndexedDataset(self.train_dataset)
        self.train_sampler = OnlineSampler(data_source=train_dataset, num_tasks=self.n_tasks, m=self.m,
                                           n=self.n, rnd_seed=self.rnd_seed, varing_NM=self.rnd_NM,num_replicas=(self.ngpus_per_nodes if self.distributed else None),rank=(self.rank if self.distributed else None))
        self.disjoint_classes = self.train_sampler.disjoint_classes
        self.disjoint_class_names = self.train_sampler.disjoint_class_names
        self.disjoint_class_num = self.train_sampler.disjoint_class_num
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.batchsize,
                                           sampler=self.train_sampler,
                                           num_workers=self.n_worker,
                                           pin_memory=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.batchsize,
                                          shuffle=False,
                                          num_workers=self.n_worker,
                                          pin_memory=True)



        logging.info(f"Incrementally training {self.n_tasks} tasks")
        task_records = defaultdict(list)
        eval_results = defaultdict(list)


        num_eval = self.eval_period

        for task_id in range(self.n_tasks):
            self.task_id = task_id
            self._total_classes = self._known_classes + self.train_sampler.disjoint_class_num[task_id]
            # 1. 在新task训练之前，用旧model提取新数据的embedding old
            if task_id > 0:
                logging.info("extract by old model start")
                train_embeddings_old, _ = self.extract_features(self.train_dataloader, self.model, None)
                logging.info("extract by old model end")
            if self.method == "joint" and task_id > 0:
                return

            logging.info("#" * 50)
            logging.info(f"# Task {task_id} Session")
            logging.info("#" * 50)
            logging.info("[2-1] Prepare a datalist for the current task")

            self.train_sampler.set_task(task_id)
            # 重新设置参数和优化器
            self.online_before_task(task_id)
            data_len = len(self.train_dataloader)
            for epoch in range(self.epochNum):
                total_loss = 0.0
                total_acc = 0.0
                samples_cnt = 0
                for i, (images, labels, idx) in enumerate(self.train_dataloader):#根据gpu数量做了拆分
                    if self.debug and (i + 1) * self.temp_batchsize >= 500:
                        break
                    samples_cnt += images.size(0) * self.world_size
                    loss, acc = self.online_step(images, labels, idx)
                    total_loss += loss
                    total_acc += acc
                self.report_training(epoch,samples_cnt, total_loss/data_len, total_acc*100/data_len)
            self.online_after_task(task_id)
            # 2. 在新task训练之后，用新model提取新数据的embedding new
            if task_id > 0:
                train_embeddings_new, _ = self.extract_features(self.train_dataloader, self.model, None)
                old_class_mean = self._class_means[:self._known_classes]
                # 3. 根据embedding old和new计算漂移量
                gap = self.displacement(train_embeddings_old, train_embeddings_new, old_class_mean, 4.0)
                del train_embeddings_old,train_embeddings_new
                gc.collect()
                if self.ssca is True:
                    old_class_mean += gap
                    self._class_means[:self._known_classes] = old_class_mean
                logging.info("tune prototype finished")
            # 4. 计算新task 新数据的prototype
            self._compute_class_mean(check_diff=False, oracle=False,task_id=task_id)#平均耗时6min
            # 重新设置 prompt token 为所有 seen class
            logging.info(f"after train1,exposed_classes:{self.exposed_classes},exposed_classes_names:{self.exposed_classes_names}")
            if self.distributed:
                self.model.module.set_prompt_token_by_clsname(self.exposed_classes_names)
            else:
                self.model.set_prompt_token_by_clsname(self.exposed_classes_names)
            # 5. 根据漂移重新训练Classifier
            if task_id > 0 and self.ca_epochs > 0 and self.ca is True:
                torch.cuda.empty_cache()
                self._stage2_compact_classifier(self.train_sampler.disjoint_class_num[task_id], self.ca_epochs)

            # 6. test
            eval_dict = self.evalue_afterTrain(task_records,task_id)
            self._known_classes = self._total_classes
        if self.is_main_process():
            np.save(os.path.join(self.log_dir, f'seed_{self.rnd_seed}.npy'),
                    task_records["task_acc"])

            if self.eval_period is not None:
                np.save(
                    os.path.join(self.log_dir,f'seed_{self.rnd_seed}_eval.npy'),
                    eval_results['test_acc'])
                np.save(
                    os.path.join(self.log_dir,f'seed_{self.rnd_seed}_eval_time.npy'),
                    eval_results['data_cnt'])
                if 'confusion_matrix' in eval_dict:
                    np.save(
                        os.path.join(self.log_dir,f'seed_{self.rnd_seed}_confusion_matrix.npy'),
                        eval_dict['confusion_matrix'])

            # Accuracy (A)
            A_auc = np.mean(eval_results["test_acc"])
            A_avg = np.mean(task_records["task_acc"])
            A_last = task_records["task_acc"][self.n_tasks - 1]

            # Forgetting (F)
            cls_acc = np.array(task_records["cls_acc"])
            acc_diff = []
            for j in range(self.n_classes):
                if np.max(cls_acc[:-1, j]) > 0:
                    acc_diff.append(np.max(cls_acc[:-1, j]) - cls_acc[-1, j])
            F_last = np.mean(acc_diff)

            logging.info(f"======== Summary =======")
            logging.info(f"Exposed Classes: {self.exposed_classes}")
            for i in range(len(cls_acc)):
                logging.info(f"Task {i}\n" + str(cls_acc[i]))
            logging.info(
                f"A_auc {A_auc:.5f} | A_avg {A_avg:.5f} | A_last {A_last:.5f} | F_last {F_last:.5f}"
            )
            with open(os.path.join(self.log_dir, 'result.txt'), 'w') as f:
                f.write(
                    f"Dataset:{self.dataset_name} | A_auc {A_auc:.5f} | A_avg {A_avg:.5f} | A_last {A_last:.5f} | F_last {F_last:.5f}\n"
                )
                f.write(f'task_acc:{task_records["task_acc"]}')

        if self.zero_shot_evaluation:
            assert hasattr(self, 'offline_evaluate')
            print("zero shot evaluation")
            for zs_dataset_name in self.zero_shot_dataset:
                zs_dataset, zs_classes_names = self.setup_zero_shot_dataset(
                    zs_dataset_name)
                zs_acc = self.offline_evaluate(zs_dataset, zs_classes_names)
                line = f"Dataset:{zs_dataset_name} | test_acc:{zs_acc:.4f}"
                print(line)
                with open(os.path.join(self.log_dir, 'result.txt'), 'a') as f:
                    f.write(line + '\n')

    def reduce_loss(self,loss, world_size):
        # 将各个GPU的损失汇总
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        return loss / world_size


    def add_new_class(self, class_name):
        exposed_classes = []
        for label in class_name:
            if label.item() not in self.exposed_classes:
                self.exposed_classes.append(label.item())
        if self.distributed:
            exposed_classes = torch.cat(
                self.all_gather(
                    torch.tensor(self.exposed_classes,
                                 device=self.device))).cpu().tolist()
            self.exposed_classes = []
            for cls in exposed_classes:
                if cls not in self.exposed_classes:
                    self.exposed_classes.append(cls)
        self.memory.add_new_class(cls_list=self.exposed_classes)

        self.exposed_classes_names = [
            self.train_dataset.classes_names[i] for i in self.exposed_classes
        ]

        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_step(self, sample, samples_cnt, idx):
        raise NotImplementedError()

    def online_before_task(self, task_id):
        raise NotImplementedError()

    def online_after_task(self, task_id):
        raise NotImplementedError()

    def online_evaluate(self, test_loader, samples_cnt):
        raise NotImplementedError()


    def evalue_afterTrain(self,task_records,task_id):
        test_sampler = OnlineTestSampler(self.test_dataset,
                                         self.exposed_classes)
        test_dataloader = DataLoader(self.test_dataset,
                                     batch_size=self.batchsize * 2,
                                     sampler=test_sampler,
                                     num_workers=self.n_worker)
        eval_dict = self.online_evaluate(test_dataloader, 1000)

        if self.distributed:
            confusion_matrix = torch.tensor(eval_dict['confusion_matrix'],
                                            device=self.device)
            eval_dict = torch.tensor([
                eval_dict['avg_loss'], eval_dict['avg_acc'],
                *eval_dict['cls_acc'], *eval_dict['task_acc']
            ],
                device=self.device)
            dist.reduce(eval_dict, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(confusion_matrix, dst=0, op=dist.ReduceOp.SUM)
            eval_dict = eval_dict.cpu().numpy()
            confusion_matrix = confusion_matrix.cpu().numpy()
            eval_dict = {
                'avg_loss': eval_dict[0] / self.world_size,
                'avg_acc': eval_dict[1] / self.world_size,
                'cls_acc': eval_dict[2:] / self.world_size,
                'task_acc': eval_dict[3:] / self.world_size,
                "confusion_matrix": confusion_matrix
            }
        task_acc = eval_dict['avg_acc']
        # ! after training done
        self.report_test(1000, eval_dict["avg_loss"], task_acc)

        logging.info("[2-4] Update the information for the current task")
        task_records["task_acc"].append(task_acc)
        task_records["cls_acc"].append(eval_dict["cls_acc"])
        if self.is_main_process() and 'confusion_matrix' in eval_dict:
            np.save(
                os.path.join(
                    self.log_dir,
                    f'seed_{self.rnd_seed}_T{task_id}_confusion_matrix.npy'
                ), eval_dict['confusion_matrix'])

        logging.info("[2-5] Report task result")
        return eval_dict


    def extract_features(self, trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (data, label, idx)= batch
                # data = data.cuda()
                # label = label.cuda()
                data = data.to(self.device)
                data = self.train_transform(data)
                label = label.to(self.device)
                embedding = self.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embedding_list, label_list

    def _compute_class_mean(self, check_diff=False, oracle=False,task_id=None):
        """
        计算当前task的class的prototype，append到所有class的prototype的list中，由 range(self._known_classes, self._total_classes)控制
        """
        if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
            ori_classes = self._class_means.shape[0]
            assert ori_classes == self._known_classes
            new_class_means = np.zeros((self._total_classes, self.feature_dim))
            new_class_means[:self._known_classes] = self._class_means
            self._class_means = new_class_means
            # new_class_cov = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self._known_classes] = self._class_covs
            self._class_covs = new_class_cov
        elif not check_diff:
            self._class_means = np.zeros((self._total_classes, self.feature_dim))  # 10,768
            # self._class_covs = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))  # 10,768,768

        radius = []
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.get_dataset_by_indices(np.arange(class_idx, class_idx + 1), source='train',
                                                                 mode='test', ret_data=True)
            if self.distributed:
                idx_sampler = DistributedSampler(idx_dataset, num_replicas=self.ngpus_per_nodes , rank=self.rank )
                idx_loader = DataLoader(idx_dataset, batch_size=self.batchsize, shuffle=False, num_workers=4,
                                        sampler=idx_sampler, pin_memory=True)
            else:
                idx_loader = DataLoader(idx_dataset, batch_size=self.batchsize, shuffle=False, num_workers=4,pin_memory=True)
            logging.info(f"class {class_idx} dataset extract start")
            vectors, _ = self._extract_vectors(idx_loader)#主要耗时是在这里，每个class要花30s
            logging.info(f"class {class_idx} dataset extract end")

            # vectors = np.concatenate([vectors_aug, vectors])

            class_mean = np.mean(vectors, axis=0)
            if task_id == 0:
                cov = np.cov(vectors.T) + np.eye(class_mean.shape[-1]) * 1e-4
                radius.append(np.trace(cov) / 768)
            # class_cov = np.cov(vectors.T)      计算协方差矩阵
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(class_mean.shape[-1]) * 1e-3

            self._class_means[class_idx, :] = class_mean
            self._class_covs[class_idx, ...] = class_cov
            del data, targets,idx_dataset
            gc.collect()

        if task_id == 0:
            self.radius = np.sqrt(np.mean(radius))
            logging.info(f"radius mean:{self.radius}")

        logging.info("_compute_class_mean finished")

        # self._class_covs.append(class_cov)

    def _extract_vectors(self, loader):
        self.model.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            _inputs = _inputs.to(self.device)
            _vectors = self.tensor2numpy(self.extract_vector(_inputs))
            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def extract_vector(self,image):
        raise NotImplementedError()


    def tensor2numpy(self, x):
        return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

    def displacement(self, Y1, Y2, embedding_old, sigma):
        DY = Y2 - Y1
        #
        distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1]) - np.tile(
            embedding_old[:, None, :], [1, Y1.shape[0], 1])) ** 2, axis=2)
        W = np.exp(-distance / (2 * sigma ** 2)) + 1e-5
        W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
        displacement = np.sum(np.tile(W_norm[:, :, None], [
            1, 1, DY.shape[1]]) * np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
        return displacement

    def _stage2_compact_classifier(self, task_size, ca_epochs=5):
        self.logit_norm = None
        for name, param in self.model.named_parameters():
            if 'prompt' in self.model_type and ("text_key" in name or "text_prompt" in name):
                param.requires_grad_(True)
            elif 'prompt' not in self.model_type and ('adaptmlp' in name or "lora" in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        # double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        logging.info(f"Parameters to be updated: {sorted(enabled)}")

        run_epochs = ca_epochs
        crct_num = self._total_classes
        param_list = [p for p in self.model.parameters() if p.requires_grad]
        network_params = [{'params': param_list, 'lr': self.lr,'weight_decay': self.wd}]
        # network_params = [{'params': param_list}]

        optimizer = optim.SGD(network_params, lr=self.lr, momentum=0.9, weight_decay=self.wd)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[4], gamma=lrate_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        # self.model.to(self._device)
        #
        # if len(self._multiple_gpus) > 1:
        #     self.model = nn.DataParallel(self.model, self._multiple_gpus)
        #即使在评估模式 (model.eval()) 下，你仍然可以进行参数训练和梯度更新。评估模式主要影响模型的行为，例如 Dropout 和 Batch Normalization 层的处理方式，但不会禁用梯度计算或优化步骤。
        self.model.eval()

        for epoch in range(run_epochs):
            losses = 0.
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 16

            for c_id in range(crct_num):
                t_id = c_id // task_size
                decay = (t_id + 1) / (self.task_id + 1) * 0.1
                # 取出class对应的prototype
                cls_mean = torch.tensor(self._class_means[c_id], dtype=torch.float64).to(self.device) * (
                        0.9 + decay)  # torch.from_numpy(self._class_means[c_id]).to(self._device)

                cls_cov = self._class_covs[c_id].to(self.device)
                # 形成正态分布
                m = MultivariateNormal(cls_mean.float(), cls_cov.float())
                # 采样特征
                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)#sampled_data_single：（8,768）
                sampled_label.extend([c_id] * num_sampled_pcls)
            # 使用采样特征再训练classifier
            sampled_data = torch.cat(sampled_data, dim=0).to(torch.float32).to(self.device)#（160,768）
            sampled_label = torch.tensor(sampled_label).long().to(self.device)
            inputs = sampled_data
            targets = sampled_label
            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for _iter in range(crct_num):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]  # 64,768
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                # 有prototype而没有prompt，则第二阶段tune的是adapter本身——这个分支废弃，因为无法在第二阶段只tune adapter
                if 'prompt' not in self.model_type and 'prototype' in self.model_type:
                    logits, _, _ = self.model(inp , image_is_feature=True)
                else:
                    # -stage two only use classifiers
                    logits,_,_ = self.model(inp, image_is_feature = True)

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self.task_id + 1):
                        cur_t_size += self.dis[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.increments[_ti]

                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)
                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)
                else:
                    loss = F.cross_entropy(logits[:, :crct_num], tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            # test_acc = self._compute_accuracy(self.model, self.test_loader)
            print('CA Task {} => Loss {:.3f}'.format(
                self.task_id, losses / self._total_classes))
        logging.info("stage2 tune prompt finishied")

    def sample_features(self,rank, world_size):
        # 根据进程 rank 设置不同的随机种子，以确保不同进程的采样不同
        torch.manual_seed(rank)
        # 这里根据特定的特征分布采样特征
        sampled_features = None  # 替换成你的特征分布采样逻辑
        return sampled_features


    def is_dist_avail_and_initialized(self):
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    def get_world_size(self):
        if not self.is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()

    def get_rank(self):
        if not self.is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()

    def is_main_process(self):
        return self.get_rank() == 0

    def setup_for_distributed(self, is_master):
        """
        This function disables printing when not in master process
        """
        self.setup_root_logger(is_master=is_master)
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)

        __builtin__.print = print

    def setup_root_logger(self, is_master=True, filename="log.txt"):
        if is_master:
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s | %(message)s")
            ch.setFormatter(formatter)
            root_logger.addHandler(ch)

            fh = logging.FileHandler(os.path.join(self.log_dir, filename),
                                     mode='w')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            root_logger.addHandler(fh)
            return root_logger
        else:
            pass

    def report_test(self, sample_num, avg_loss, avg_acc):
        logging.info(
            f"Test | Sample # {sample_num} | test_loss {avg_loss:.2f} | test_acc {avg_acc:.2f} | "
        )

    def report_training(self, epoch,sample_num, train_loss, train_acc):
        total_num = torch.tensor(sample_num).cuda(self.gpu)
        sample_num = self.reduce_loss(total_num,self.world_size)
        avg_loss ,avg_acc= torch.tensor(train_loss, dtype=torch.float32).cuda(self.gpu),torch.tensor(train_acc, dtype=torch.float32).cuda(self.gpu)
        train_loss ,train_acc=  self.reduce_loss(avg_loss,self.world_size),self.reduce_loss(avg_acc,self.world_size)
        logging.info(
            f"Train | epoch:{epoch}, Sample # {sample_num} | train_loss {train_loss:.3f} | train_acc {train_acc:.2f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def _interpret_pred(self, y, pred):
        # xlable is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

    def all_gather(self, item):
        local_size = torch.tensor(item.size(0), device=self.device)
        all_sizes = [
            torch.zeros_like(local_size) for _ in range(dist.get_world_size())
        ]
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.gather(local_size, all_sizes, dst=i)
            else:
                dist.gather(local_size, dst=i)
        # dist.all_gather(all_sizes, local_size, async_op=False)
        max_size = max(all_sizes)

        size_diff = max_size.item() - local_size.item()
        if size_diff:
            padding = torch.zeros(size_diff,
                                  device=self.device,
                                  dtype=item.dtype)
            item = torch.cat((item, padding))

        all_qs_padded = [
            torch.zeros_like(item) for _ in range(dist.get_world_size())
        ]

        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.gather(item, all_qs_padded, dst=i)
            else:
                dist.gather(item, dst=i)

        # dist.all_gather(all_qs_padded, item)
        all_qs = []
        for q, size in zip(all_qs_padded, all_sizes):
            all_qs.append(q[:size])
        return all_qs

    def train_data_config(self, n_task, train_dataset, train_sampler):
        for t_i in range(n_task):
            train_sampler.set_task(t_i)
            train_dataloader = DataLoader(train_dataset,
                                          batch_size=self.batchsize,
                                          sampler=train_sampler,
                                          num_workers=4)
            data_info = {}
            for i, data in enumerate(train_dataloader):
                _, label = data
                label = label.to(self.device)
                for b in range(len(label)):
                    if 'Class_' + str(label[b].item()) in data_info.keys():
                        data_info['Class_' + str(label[b].item())] += 1
                    else:
                        data_info['Class_' + str(label[b].item())] = 1
            logging.info(f"[Train] Task{t_i} Data Info")
            logging.info(data_info)

            convert_data_info = self.convert_class_label(data_info)
            np.save(
                os.path.join(self.log_dir,
                             f"seed_{self.rnd_seed}_task{t_i}_train_data.npy"),
                convert_data_info)
            logging.info(convert_data_info)

    def test_data_config(self, test_dataloader, task_id):
        data_info = {}
        for i, data in enumerate(test_dataloader):
            _, label = data
            label = label.to(self.device)

            for b in range(len(label)):
                if 'Class_' + str(label[b].item()) in data_info.keys():
                    data_info['Class_' + str(label[b].item())] += 1
                else:
                    data_info['Class_' + str(label[b].item())] = 1

        logging.info("<<Exposed Class>>")
        logging.info([
            (x, y)
            for x, y in zip(self.exposed_classes, self.exposed_classes_names)
        ])

        logging.info(f"[Test] Task {task_id} Data Info")
        logging.info(data_info)
        logging.info("<<Convert>>")
        convert_data_info = self.convert_class_label(data_info)
        logging.info(convert_data_info)

    def convert_class_label(self, data_info):
        #* self.class_list => original class label
        self.class_list = self.train_dataset.classes
        for key in list(data_info.keys()):
            old_key = int(key[6:])
            data_info[self.class_list[old_key]] = data_info.pop(key)

        return data_info

    def current_task_data(self, train_loader):
        data_info = {}
        for i, data in enumerate(train_loader):
            _, label = data

            for b in range(label.shape[0]):
                if 'Class_' + str(label[b].item()) in data_info.keys():
                    data_info['Class_' + str(label[b].item())] += 1
                else:
                    data_info['Class_' + str(label[b].item())] = 1

        logging.info("Current Task Data Info")
        logging.info(data_info)
        logging.info("<<Convert to str>>")
        convert_data_info = self.convert_class_label(data_info)
        logging.info(convert_data_info)
