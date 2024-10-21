import gc
import random
import time
import logging
import datetime
import os.path as osp
from tqdm import tqdm
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader, DistributedSampler
from PIL import Image

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from methods._trainer import _Trainer
from utils.train_utils import select_optimizer, select_scheduler
from utils.memory import MemoryBatchSampler

logger = logging.getLogger()


class Proto_CLIP(_Trainer):

    def __init__(self, **kwargs):
        super(Proto_CLIP, self).__init__(**kwargs)
        self.batch_exposed_classes = []
        self.batch_exposed_classes_names = []
        self.visible_classes = self.args.get('visible_classes', 'batch')


    def online_step(self, images, labels, idx):
        self.add_new_class(labels)#将新出现的classname加入到self.exposed_class和_names变量中
        if self.distributed:
            self.model.module.update_class_names(self.exposed_classes_names)#将新出现的classname加入到self.current_class_names变量中
        else:
            self.model.update_class_names(self.exposed_classes_names)

        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1
        return _loss / _iter, _acc / _iter

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0

        if self.visible_classes == 'batch':#这个
            # batch
            train_class_list = self.batch_exposed_classes
            train_class_name_list = self.batch_exposed_classes_names

        else:
            # all
            train_class_list = self.exposed_classes
            train_class_name_list = self.exposed_classes_names

        x, y = data

        if len(self.memory) > 0 and self.memory_batchsize > 0:
            memory_images, memory_labels = next(self.memory_provider)
            for i in memory_labels.unique():
                if i not in train_class_list:
                    train_class_list.append(i)
                    train_class_name_list.append(self.exposed_classes_names[
                        self.exposed_classes.index(i)])
            x = torch.cat([x, memory_images], dim=0)
            y = torch.cat([y, memory_labels], dim=0)

        for j in range(len(y)):#只用batch class做训练,所以将标签替换
            y[j] = train_class_list.index(y[j].item())
        x = x.to(self.device)
        y = y.to(self.device)

        x = self.train_transform(x)
        # 只用当前batch的classname来做train
        if self.distributed:
            self.model.module.set_prompt_token_by_clsname(train_class_name_list)
        else:
            self.model.set_prompt_token_by_clsname(train_class_name_list)

        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit, image_features, text_features = self.model(x)
            loss = self.criterion(logit, y)
        _, preds = logit.topk(self.topk, 1, True, True)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        if self.args.get('grad_analysis', False):
            self._grad_analysis(image_features.clone().detach(),
                                text_features.clone().detach(),
                                y.clone().detach(), train_class_list)

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct / total_num_data

    def extract_vector(self,image):
        if self.distributed:
            image_features = self.model.module.encode_image(image)#image:(32,3,32,32)
        else:
            image_features = self.model.encode_image(image)#image:(32,3,32,32)
        # image_features = self.model(image,encode_image=True)  # image:(32,3,32,32)
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features


    def online_before_task(self, task_id):
        # Freeze some parameters
        for k, v in self.model.named_parameters():
            if "adaptmlp" in k or "lora" in k or "text_key" in k or "text_prompt" in k:
                v.requires_grad = True
            else:
                v.requires_grad = False

        logger.info("Total parameters:\t{}".format(
            sum(p.numel() for p in self.model.parameters())))
        logger.info("Trainable parameters:\t{}".format(
            sum(p.numel() for p in self.model.parameters()
                if p.requires_grad)))
        # double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        logging.info(f"Parameters to be updated: {sorted(enabled)}")
        self.reset_opt()

        self.compute_old_embedding()



    def compute_old_embedding(self):
        # 1. 在新task训练之前，用旧model提取新数据的embedding old
        if self.task_id > 0:
            logging.info("extract by old model start")
            self.train_embeddings_old, _ = self.extract_features(self.train_dataloader, self.model, None)
            logging.info("extract by old model end")

    def online_after_task(self, task_id):
        self.stage1_and_stage2()


    def stage1_and_stage2(self):
        # 2. 在新task训练之后，用新model提取新数据的embedding new
        if self.task_id > 0:
            train_embeddings_new, _ = self.extract_features(self.train_dataloader, self.model, None)
            old_class_mean = self._class_means[:self._known_classes]
            # 3. 根据embedding old和new计算漂移量
            gap = self.displacement(self.train_embeddings_old, train_embeddings_new, old_class_mean, 4.0)
            del self.train_embeddings_old, train_embeddings_new
            gc.collect()
            if self.ssca is True:
                old_class_mean += gap
                self._class_means[:self._known_classes] = old_class_mean
            logging.info("tune prototype finished")
        # 4. 计算新task 新数据的prototype
        self._compute_class_mean(check_diff=False, oracle=False, task_id=self.task_id)  # 平均耗时6min

        # 5. 根据漂移重新训练Classifier
        if self.task_id > 0 and self.ca_epochs > 0 and self.ca is True:
            torch.cuda.empty_cache()
            if self.distributed:
                self.model.module.set_prompt_token_by_clsname(self.all_classnames[:self._total_classes])
                # self.model.module.set_prompt_token_by_clsname(self.all_classnames[:self._known_classes])
            else:
                self.model.set_prompt_token_by_clsname(self.all_classnames[:self._total_classes])
                # self.model.set_prompt_token_by_clsname(self.all_classnames[:self._known_classes])
            self._stage2_compact_classifier(self.train_sampler.disjoint_class_num[self.task_id], self.ca_epochs)
        # 重新设置 prompt token 为所有 seen class
        logging.info(
            f"after stage 1,known_classes:{self.classes[:self._total_classes]},known_classes_names:{self.all_classnames[:self._total_classes]}")
        if self.distributed:
            self.model.module.set_prompt_token_by_clsname(self.all_classnames[:self._total_classes])
        else:
            self.model.set_prompt_token_by_clsname(self.all_classnames[:self._total_classes])

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
            del vectors,data, targets,idx_dataset,idx_loader
            gc.collect()

        if task_id == 0:
            self.radius = np.sqrt(np.mean(radius))
            logging.info(f"radius mean:{self.radius}")

        logging.info("_compute_class_mean finished")

        # self._class_covs.append(class_cov)

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


    def _extract_vectors(self, loader):
        self.model.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            _inputs = _inputs.to(self.device)
            _vectors = self.tensor2numpy(self.extract_vector(_inputs))
            vectors.append(_vectors)
            targets.append(_targets)
            del _inputs
            gc.collect()

        return np.concatenate(vectors), np.concatenate(targets)

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
        crct_num = self._known_classes
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
            num_sampled_pcls = 8

            for c_id in range(crct_num):
                t_id = c_id // task_size
                decay = (t_id + 1) / (self.task_id + 1) * 0.1
                # 取出class对应的prototype
                cls_mean = torch.tensor(self._class_means[c_id], dtype=torch.float64) * (
                        0.9 + decay)  # torch.from_numpy(self._class_means[c_id]).to(self._device)

                cls_cov = self._class_covs[c_id]
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
                del inp,tgt
                gc.collect()

            scheduler.step()
            # test_acc = self._compute_accuracy(self.model, self.test_loader)
            print('CA Task {} => Loss {:.3f}'.format(
                self.task_id, losses / self._total_classes))
        logging.info("stage2 tune prompt finishied")


    def online_evaluate(self, test_loader, samples_cnt):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        total_correct1 = 0.0
        correct_l = torch.zeros(self.n_tasks)
        num_data_l = torch.zeros(self.n_tasks)
        label = []
        pred_list = []
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                # for j in range(len(y)):#标签不能按exposed
                #     y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                logit, _, _ = self.model(x,test=True)
                pred = torch.argmax(logit, dim=-1)
                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().item()
                num_data_l += xlabel_cnt.detach().item()

                _, preds = logit.topk(self.topk, 1, True, True)#topk=1
                total_correct += torch.sum(correct_xlabel_cnt)
                total_correct1 += torch.sum(preds == y.unsqueeze(1)).item()
                logging.info(f"correct:{total_correct}  {total_correct1}")


                label += y.tolist()
                pred_list += pred.tolist()

        avg_acc = torch.sum(correct_l) / torch.sum(num_data_l)
        avg_loss = total_loss / torch.sum(num_data_l)
        task_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        # 打印每个任务的正确率
        logging.info(f'task_acc:{task_acc}')
        # print(f'task_acc:{task_acc}')

        cm = confusion_matrix(label, pred_list)

        eval_dict = {
            "avg_loss": avg_loss,
            "avg_acc": avg_acc,
            "cls_acc": task_acc,
            "task_acc": task_acc,
            "confusion_matrix": cm.tolist()
        }
        return eval_dict

    def _accuracy_mean_task(self, logits):
        acc_per_task = [0 for _ in range(ses + 1)]
        count_per_task = [0 for _ in range(ses + 1)]
        for i, (x, y) in enumerate(loader):
            self.model.eval()
            pred_y = self.inference(x.cuda(), y.cuda(), ses, test_class)
            _, top_labels = pred_y.topk(1, dim=-1)
            for t in range(ses + 1):
                acc_per_task[t] += ((top_labels.view(-1) == y.cuda()) * (y.cuda() // self.args.class_per_task == t)).sum().item()
                count_per_task[t] += (y.cuda() // self.args.class_per_task == t).sum().item()
        acc = [a * 1.0 / c for (a, c) in zip(acc_per_task, count_per_task)]
        # acc = np.array(acc).mean()

        return acc


    def offline_evaluate(self, test_loader, classes_names):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label, pred_list = [], []

        text_tokens = self.model.labels_tokenize(classes_names)
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(test_loader):
                x, y = data

                x = x.to(self.device)
                y = y.to(self.device)

                logit, _, _ = self.model(x, text_tokens)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                label += y.tolist()
                pred_list += pred.tolist()

        total_acc = total_correct / total_num_data

        return total_acc

    def update_memory(self, sample, label):
        # Update memory
        if self.distributed:
            sample = torch.cat(self.all_gather(sample.to(self.device)))
            label = torch.cat(self.all_gather(label.to(self.device)))
            sample = sample.cpu()
            label = label.cpu()
        idx = []
        if self.is_main_process():
            for lbl in label:
                self.seen += 1
                if len(self.memory) < self.memory_size:
                    idx.append(-1)
                else:
                    j = torch.randint(0, self.seen, (1, )).item()
                    if j < self.memory_size:
                        idx.append(j)
                    else:
                        idx.append(self.memory_size)
        # Distribute idx to all processes
        if self.distributed:
            idx = torch.tensor(idx).to(self.device)
            size = torch.tensor([idx.size(0)]).to(self.device)
            dist.broadcast(size, 0)
            if dist.get_rank() != 0:
                idx = torch.zeros(size.item(),
                                  dtype=torch.long).to(self.device)
            dist.barrier()  # wait for all processes to reach this point
            dist.broadcast(idx, 0)
            idx = idx.cpu().tolist()
        # idx = torch.cat(self.all_gather(torch.tensor(idx).to(self.device))).cpu().tolist()
        for i, index in enumerate(idx):
            if len(self.memory) >= self.memory_size:
                if index < self.memory_size:
                    self.memory.replace_data([sample[i], label[i].item()],
                                             index)
            else:
                self.memory.replace_data([sample[i], label[i].item()])

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer,
                                              None)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer,
                                          None)

    def add_new_batch_class(self, class_name):
        batch_exposed_classes = []
        for label in class_name:
            if label.item() not in self.batch_exposed_classes:
                self.batch_exposed_classes.append(label.item())
        if self.distributed:
            batch_exposed_classes = torch.cat(
                self.all_gather(
                    torch.tensor(self.batch_exposed_classes,
                                 device=self.device))).cpu().tolist()
            self.batch_exposed_classes = []
            for cls in batch_exposed_classes:
                if cls not in self.batch_exposed_classes:
                    self.batch_exposed_classes.append(cls)
        self.batch_exposed_classes_names = [
            self.train_dataset.classes_names[i]
            for i in self.batch_exposed_classes
        ]

    def add_new_class(self, class_name):
        _old_num = len(self.exposed_classes)
        super().add_new_class(class_name)#将这批数据的新class加入到exposed_class变量中

        self.batch_exposed_classes = []
        self.batch_exposed_classes_names = []
        if self.memory_size > 0:
            self.batch_exposed_classes = self.exposed_classes
            self.batch_exposed_classes_names = self.exposed_classes_names
        else:
            self.add_new_batch_class(class_name)#再将这批数据的新class从exposed_class变量中取出，加入到batch_exposed_classes变量中

    def report_training(self, epoch,sample_num, train_loss, train_acc):
        logging.info(
            f"Train | epoch:{epoch}, Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"Num_Batch_Classes {len(self.batch_exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

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