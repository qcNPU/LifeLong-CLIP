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
logging.basicConfig(level=logging.INFO,  # 设置日志级别
                    format='%(asctime)s - %(levelname)s - %(message)s')


##################################################################
# This is trainer with a DistributedDataParallel                 #
# Based on the following tutorial:                               #
# https://github.com/pytorch/examples/blob/main/imagenet/main.py #
# And Deit by FaceBook                                           #
# https://github.com/facebookresearch/deit                       #
##################################################################
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
        self.peft_encoder = kwargs.get("peft_encoder")
        self.num_sampled_pcls = kwargs.get("num_sampled_pcls")
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
        self.world_size = 1
        if "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"] != '':
            self.world_size = int(
                os.environ["WORLD_SIZE"]) * self.ngpus_per_nodes
        else:
            self.world_size = self.world_size * self.ngpus_per_nodes#4
        # self.distributed = self.world_size > 1
        # if self.distributed:
        #     self.batchsize = self.batchsize // self.world_size#64,4,16
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
        # self.device = torch.device(self.gpu_ids[0])
        # self.custom_clip = self.custom_clip.to(self.device)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.criterion =  nn.CrossEntropyLoss(reduction="mean")
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.custom_clip)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)
        if self.distributed:
            self.custom_clip = nn.DataParallel(self.custom_clip)
        self.custom_clip = self.custom_clip.cuda()

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
        self.gpu_ids = list(range(self.ngpus_per_nodes))
        self.distributed = self.ngpus_per_nodes>1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.custom_clip, self.inp_size = get_model(
            model_name=self.model_name,
            method=self.method,
            num_classes=self.n_classes,
            device=torch.device('cpu'),#这里设置加载device
            peft_encoder=self.args['peft_encoder'],
            args = argparse.Namespace(**self.args)
        )

        self.dtype = self.custom_clip.dtype

        self.setup_transforms()
        self.setup_dataset()
        self.setup_distributed_model()
        self.memory = Memory()
        self.total_samples = len(self.train_dataset)

        train_dataset = IndexedDataset(self.train_dataset)
        self.train_sampler = OnlineSampler(data_source=train_dataset, num_tasks=self.n_tasks, m=self.m,
                                           n=self.n, rnd_seed=self.rnd_seed, varing_NM=self.rnd_NM)
        self.disjoint_classes = self.train_sampler.disjoint_classes
        self.disjoint_class_names = self.train_sampler.disjoint_class_names
        self.all_classnames = self.train_sampler.class_names
        self.classes = self.train_sampler.classes
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
                    samples_cnt += images.size(0)
                    loss, acc = self.online_step(images, labels, idx)
                    total_loss += loss
                    total_acc += acc
                self.report_training(epoch,samples_cnt, total_loss/data_len, total_acc*100/data_len)
            self.online_after_task(task_id)

            # 6. test
            eval_dict = self.evalue_afterTrain(task_records,task_id)
            self._known_classes = self._total_classes
        self.save_result(task_records,eval_results,eval_dict)

    def save_result(self,task_records,eval_results,eval_dict):
        np.save(os.path.join(self.log_dir, f'seed_{self.rnd_seed}.npy'), task_records["task_acc"])
        if self.eval_period is not None:
            np.save(os.path.join(self.log_dir,f'seed_{self.rnd_seed}_eval.npy'), eval_results['test_acc'])
            np.save(os.path.join(self.log_dir,f'seed_{self.rnd_seed}_eval_time.npy'), eval_results['data_cnt'])
            if 'confusion_matrix' in eval_dict:
                np.save(os.path.join(self.log_dir,f'seed_{self.rnd_seed}_confusion_matrix.npy'), eval_dict['confusion_matrix'])

        # Accuracy (A)
        A_auc = np.mean(eval_results["test_acc"])
        A_avg = np.mean(task_records["task_acc"])
        A_last = task_records["task_acc"][self.n_tasks - 1]

        # Forgetting (F)
        cls_acc = np.array(task_records["cls_acc"])
        acc_diff = []
        for j in range(self.n_tasks):
            if np.max(cls_acc[:-1, j]) > 0:
                acc_diff.append(np.max(cls_acc[:-1, j]) - cls_acc[-1, j])
        F_last = np.mean(acc_diff)

        logging.info(f"======== Summary =======")
        logging.info(f"Exposed Classes: {self.exposed_classes}")
        for i in range(len(cls_acc)):
            logging.info(f"Task {i}:" + str(cls_acc[i]))
        logging.info(f"Task avg:"+str(task_records["task_acc"]))
        logging.info(f"A_avg {A_avg:.5f} | A_last {A_last:.5f} | F_last {F_last:.5f}")
        with open(os.path.join(self.log_dir, 'result.txt'), 'w') as f:
            f.write(f"Dataset:{self.dataset_name} | A_auc {A_auc:.5f} | A_avg {A_avg:.5f} | A_last {A_last:.5f} | F_last {F_last:.5f}\n")
            f.write(f'task_acc:{task_records["task_acc"]}\n')
            f.write(f'per_task_acc:{task_records["cls_acc"]}')

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


    def add_new_class(self, class_name):
        exposed_classes = []
        for label in class_name:
            if label.item() not in self.exposed_classes:
                self.exposed_classes.append(label.item())
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

        task_acc = eval_dict['avg_acc']
        # ! after training done
        self.report_test(1000, eval_dict["avg_loss"], task_acc)

        logging.info("[2-4] Update the information for the current task")
        task_records["task_acc"].append(task_acc)
        task_records["cls_acc"].append(eval_dict["cls_acc"])
        logging.info("[2-5] Report task result")
        return eval_dict

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
        logging.info(
            f"Train | epoch:{epoch}, Sample # {sample_num} | train_loss {train_loss:.3f} | train_acc {train_acc:.2f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def _interpret_pred(self, y, pred):
        # xlable is batch
        ret_num_data = torch.zeros(10)
        ret_corrects = torch.zeros(10)
        cls = y // self.n_tasks
        xlabel_cls, xlabel_cnt = cls.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        cls_correct = correct_xlabel // self.n_tasks
        correct_cls, correct_cnt = cls_correct.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.custom_clip)
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
