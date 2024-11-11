import gc
import random
import time
import logging
import datetime
import os.path as osp
from tqdm import tqdm
import os

import numpy as np
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from methods._trainer import _Trainer
from utils.train_utils import select_optimizer, select_scheduler
from utils.memory import MemoryBatchSampler
from datasets.gpt.gpt_generation import attributes
from sklearn.cluster import KMeans

logger = logging.getLogger()


class Trainer_ProtoCLIP(_Trainer):

    def __init__(self, **kwargs):
        super(Trainer_ProtoCLIP, self).__init__(**kwargs)
        self.overall_key_counts = None
        self.key_statis = None
        self.cls_en_map = None
        self.cluster_info = None
        self.batch_exposed_classes = []
        self.batch_exposed_classes_names = []
        self.visible_classes = self.args.get('visible_classes', 'batch')

    def before_train(self):
        # 获取数据集所有class name的Attribute的word embedding
        cls_en_map = self.getTaskAttributeEmbedding(args=self.args, class_names=self.all_classnames,
                                                    clip_model=self.custom_clip.module,
                                                    text_encoder=self.custom_clip.module.text_encoder)
        # self.custom_clip.module.init_cls_map(cls_en_map)
        # self.custom_clip.module.cls_en_map = cls_en_map
        self.custom_clip.module.all_classnames = self.all_classnames
        cluster_info = self.cluster_attributes(cls_en_map)
        self.custom_clip.module.cluster_info = cluster_info



    def online_before_task(self, task_id):
        # 这里传custom_clip.module和custom_clip是一样的，会自动选择到module里的参数
        if self.distributed:
            model = self.custom_clip.module
        else:
            model = self.custom_clip
        model._known_classes = self._known_classes
        model._total_classes = self._total_classes
        # Freeze some parameters
        for k, v in model.named_parameters():
            if "adaptmlp" in k or "lora" in k or "text_key" in k or "text_prompt" in k:
                v.requires_grad = True
            else:
                v.requires_grad = False

        logger.info("Total parameters:\t{}".format(sum(p.numel() for p in model.parameters())))
        logger.info("Trainable parameters:\t{}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        # double check
        enabled = set()
        for name, param in model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        logging.info(f"Parameters to be updated: {sorted(enabled)}")
        self.reset_opt(model)
        self.compute_old_embedding()

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)  # 将新出现的classname加入到self.exposed_class和_names变量中
        if self.distributed:
            self.custom_clip.module.update_class_names(
                self.exposed_classes_names)  # 将新出现的classname加入到self.current_class_names变量中
        else:
            self.custom_clip.update_class_names(self.exposed_classes_names)

        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1
        return _loss / _iter, _acc / _iter

    def online_train(self, data):
        self.custom_clip.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0

        if self.visible_classes == 'batch':  # 这个
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

        # x = x.to(self.device)
        # y = y.to(self.device)
        x = x.cuda()
        y = y.cuda()

        x = self.train_transform(x)
        # 只用当前batch的classname来做train
        if self.distributed:
            self.custom_clip.module.train_class_list = train_class_list
            self.custom_clip.module.set_prompt_token_by_clsname(classnames=train_class_name_list)
        else:
            self.custom_clip.set_prompt_token_by_clsname(classnames=train_class_name_list)

        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit, reg_logits,image_features, template_fea,selected_key = self.custom_clip(image=x, labels=y)
            # 只用batch class做CE
            for j in range(len(y)):
                y[j] = train_class_list.index(y[j].item())
            loss_ce = self.criterion(logit, y)
            loss_reg = self.criterion(reg_logits,y)
            loss_key = self.cosine_loss(image_features,selected_key)
            loss = loss_ce + 20*loss_reg + loss_key
        with open(os.path.join(self.log_dir, 'loss.txt'), 'w') as f:
            f.write(f"ce:{loss_ce} | reg:{loss_reg} | key:{loss_key}\n")
        _, preds = logit.topk(self.topk, 1, True, True)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        # if self.args.get('grad_analysis', False):
        #     self._grad_analysis(image_features.clone().detach(),
        #                         text_features.clone().detach(),
        #                         y.clone().detach(), train_class_list)

        total_loss += loss.detach().item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct / total_num_data

    def cosine_loss(self,q, k):
        # pdb.set_trace()
        q = q.repeat(1, k.shape[1], 1)
        # k = k.squeeze(1)
        # q = q/q.norm(dim=-1)
        k_norm = k.norm(dim=-1, keepdim=True)
        # pdb.set_trace()
        # k_norm = k.norm(dim=-1).unsqueeze(1).repeat(1,k.shape[1])
        k = k / k_norm
        cos = ((q * k) / (k.shape[0] * k.shape[1])).sum()
        return 1 - cos


    def online_after_task(self, task_id):
        self.stage1_and_stage2()

    def online_evaluate(self, test_loader, samples_cnt):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        total_correct1 = 0.0
        correct_l = torch.zeros(self.n_tasks)
        num_data_l = torch.zeros(self.n_tasks)
        label = []
        pred_list = []
        self.custom_clip.eval()

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                # for j in range(len(y)):#标签不能按exposed
                #     y[j] = self.exposed_classes.index(y[j].item())

                x = x.cuda()
                y = y.cuda()

                logit = self.custom_clip(x, test=True)
                pred = torch.argmax(logit, dim=-1)
                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt
                num_data_l += xlabel_cnt
                del x, y
                gc.collect()
                # label += y.tolist()
                # pred_list += pred.tolist()

            avg_acc = torch.sum(correct_l) * 100 / torch.sum(num_data_l)
            avg_loss = total_loss / torch.sum(num_data_l)
            task_acc = np.around((correct_l * 100 / num_data_l).numpy().tolist(), 2)

        # 打印每个任务的正确率
        logging.info(f'task_acc:{task_acc}')

        # cm = confusion_matrix(label, pred_list)

        eval_dict = {
            "avg_loss": avg_loss,
            "avg_acc": avg_acc,
            "cls_acc": task_acc,
            "task_acc": task_acc,
            # "confusion_matrix": cm.tolist()
            "confusion_matrix": []
        }
        return eval_dict

    def extract_vector(self, image):
        if self.distributed:
            image_features = self.custom_clip.module.encode_image(image)  # image:(32,3,32,32)
        else:
            image_features = self.custom_clip.encode_image(image)  # image:(32,3,32,32)
        # image_features = self.model(image,encode_image=True)  # image:(32,3,32,32)
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def compute_old_embedding(self):
        # 1. 在新task训练之前，用旧model提取新数据的embedding old
        if self.task_id > 0:
            logging.info("extract by old model start")
            self.train_embeddings_old, _ = self.extract_features(self.train_dataloader, self.custom_clip, None)
            logging.info("extract by old model end")

    def stage1_and_stage2(self):
        # 2. 在新task训练之后，用新model提取新数据的embedding new
        if self.task_id > 0:
            train_embeddings_new, _ = self.extract_features(self.train_dataloader, self.custom_clip, None)
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

        # 重新设置 prompt token 为所有 seen class
        logging.info(f"after stage 1,total_classes_names:{self.all_classnames[:self._total_classes]}")
        if self.distributed:
            self.custom_clip.module.set_prompt_token_by_clsname(self.all_classnames[:self._total_classes])
        else:
            self.custom_clip.set_prompt_token_by_clsname(self.all_classnames[:self._total_classes])
        # 5. 根据漂移重新训练Classifier
        if self.task_id > 0 and self.ca_epochs > 0 and self.ca is True:
            torch.cuda.empty_cache()
            self._stage2_compact_classifier(self.train_sampler.disjoint_class_num[self.task_id], self.ca_epochs)

    def extract_features(self, trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (data, label, idx) = batch
                # data = data.cuda()
                # label = label.cuda()
                data = data.cuda()
                data = self.train_transform(data)
                label = label.cuda()
                embedding = self.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embedding_list, label_list

    def _compute_class_mean(self, check_diff=False, oracle=False, task_id=None):
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
        logging.info(f"class mean  extract start")
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.get_dataset_by_indices(np.arange(class_idx, class_idx + 1),
                                                                     source='train',
                                                                     mode='test', ret_data=True)

            idx_loader = DataLoader(idx_dataset, batch_size=self.batchsize, shuffle=False, num_workers=4,
                                    pin_memory=True)
            vectors, _ = self._extract_vectors(idx_loader)  # 主要耗时是在这里，每个class要花30s

            # vectors = np.concatenate([vectors_aug, vectors])

            class_mean = np.mean(vectors, axis=0)
            if task_id == 0:
                cov = np.cov(vectors.T) + np.eye(class_mean.shape[-1]) * 1e-4
                radius.append(np.trace(cov) / 768)
            # class_cov = np.cov(vectors.T)      计算协方差矩阵
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(class_mean.shape[-1]) * 1e-3

            self._class_means[class_idx, :] = class_mean
            self._class_covs[class_idx, ...] = class_cov
            del data, targets, idx_dataset
            gc.collect()
        logging.info(f"class mean extract end")
        if task_id == 0:
            self.radius = np.sqrt(np.mean(radius))
            logging.info(f"radius mean:{self.radius}")

        logging.info("_compute_class_mean finished")

        # self._class_covs.append(class_cov)

    def getTaskAttributeEmbedding(self, args, class_names, clip_model, text_encoder):
        # cls_str_map = getTaskEntitys(args, class_names)
        cls_str_map = self.getTaskAttributes(args, class_names)
        attr_token_map = {}
        attr_fea_map = {}
        with torch.no_grad():
            for index, attrs in cls_str_map.items():
                # 遇到DataParallel’ object has no attribute ‘xxxx’时，在model后面加上.module.
                tokenized_keys = torch.cat([clip_model.tokenize(p).cuda() for p in attrs])  # （298,77）
                attr_token = clip_model.token_embedding(tokenized_keys)
                attr_token_map[index] = attr_token
                attr_fea = text_encoder(text=tokenized_keys, tokenized_prompts=None, need_token=True)
                # entity_embeddings,_ = entity_embeddings.max(dim=1)
                attr_fea /= attr_fea.norm(dim=-1, keepdim=True)  # 归一化（298,768）
                attr_fea_map[index] = attr_fea

        return [cls_str_map, attr_token_map, attr_fea_map]

    def getTaskAttributes(self, args, train_classnames):
        # 取出task中所有class的entity和attribute，合并去重
        class_attributes = attributes.get_Classes_Attributes(args, train_classnames)
        classMap = {}
        for i, info in enumerate(class_attributes):
            attrs = list()
            for j in info:
                a1 = [s for s in j.split("|") if s.strip() != '']
                attrs.extend(a1)
            classMap[i] = attrs
        return classMap

    # def init_cls_map(self, cls_en_map):
    #     self.cls_en_map = cls_en_map
    #     self.cluster_info = self.cluster_attributes(cls_en_map)
    #     self.key_statis = {i: {j: 0 for j in range(10)} for i in range(self.args.class_per_task * (self.args.sess + 1))}
    #     self.overall_key_counts = {cls: {i: 0 for i in range(self.args.num_prompt)} for cls in range(100)}

    def cluster_attributes(self, cls_en_map):
        num_clusters = 3
        max_iterations = 100
        cluster_features = []
        cluster_strs = []
        cluster_tokens = []
        tolerance = 1e-4
        for ind, emb in cls_en_map[2].items():
            # 使用 kmeans-pytorch 进行 K-means 聚类
            # cluster_ids_x, cluster_centers = kmeans(
            #     X=emb, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda')
            # )

            kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iterations, n_init=10, tol=tolerance, random_state=42)
            kmeans.fit(emb.cpu().numpy())  # 使用 numpy 数据

            # 获取聚类分配结果
            cluster_ids_x = kmeans.labels_
            # 根据聚类分配结果将样本分到不同的组
            features = [[] for _ in range(num_clusters)]
            strs = [[] for _ in range(num_clusters)]
            tokens = [[] for _ in range(num_clusters)]
            for i, cluster_id in enumerate(cluster_ids_x):  # i是索引，cluster_id是它属于哪个簇
                features[cluster_id].append(emb[i])
                strs[cluster_id].append(cls_en_map[0][ind][i])
                # tokens[cluster_id].append(cls_en_map[1][ind][i])
            features = [torch.stack(i,dim=0) for i in features]
            cluster_features.append(features)
            cluster_strs.append(strs)
            cluster_tokens.append(strs)

        return [cluster_strs, cluster_tokens, cluster_features]

    def image_display(images, attributes, prefix, grid_size=(3, 3)):
        """
        显示图片网格及其对应的 Attribute 字符串

        :param images: 图片的张量列表
        :param attributes: 图片对应的 Attribute 字符串列表
        :param grid_size: 网格的行列数 (rows, cols)
        """
        nrow = 4
        # 将张量转换为 numpy 数组并归一化到 [0, 1]
        image_np = (images - images.min()) / (images.max() - images.min())
        num_images = len(images)
        ncol = (num_images + nrow - 1) // nrow
        fig, axes = plt.subplots(ncol, nrow, figsize=(32, 22))

        for i in range(num_images):
            row = i // nrow
            col = i % nrow
            ax = axes[row, col]
            img = image_np[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            ax.set_title("\n".join(attributes[i]), fontsize=20)
            ax.axis('off')

        # Hide any unused subplots
        for i in range(num_images, ncol * nrow):
            fig.delaxes(axes.flat[i])

        plt.subplots_adjust(wspace=0.4, hspace=0.8)
        fig.savefig(f"{prefix}.png")
        plt.close(fig)

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
        self.custom_clip.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            _inputs = _inputs.cuda()
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
        lr = 5e-3
        self.logit_norm = None
        model = self.custom_clip.module if self.distributed else self.custom_clip
        isprompt = 'prompt' in self.model_type
        isboth = self.peft_encoder == 'both'
        for name, param in model.named_parameters():
            if isprompt and ("text_key" in name or "text_prompt" in name):
                param.requires_grad = True
            else:
                if isboth and "adaptmlp" in name and "model.transformer" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        # double check
        enabled = set()
        for name, param in model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        logging.info(f"Parameters to be updated: {sorted(enabled)}")

        run_epochs = ca_epochs
        crct_num = self._total_classes
        param_list = [p for p in model.parameters() if p.requires_grad]
        network_params = [{'params': param_list, 'lr': lr, 'weight_decay': self.wd}]
        # network_params = [{'params': param_list}]

        optimizer = optim.SGD(network_params, lr=lr, momentum=0.9, weight_decay=self.wd)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[4], gamma=lrate_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        # 即使在评估模式 (model.eval()) 下，你仍然可以进行参数训练和梯度更新。评估模式主要影响模型的行为，例如 Dropout 和 Batch Normalization 层的处理方式，但不会禁用梯度计算或优化步骤。
        self.custom_clip.eval()

        cls_normals = {}
        for c_id in range(crct_num):
            t_id = c_id // task_size
            decay = (t_id + 1) / (self.task_id + 1) * 0.1
            # 取出class对应的prototype
            cls_mean = torch.tensor(self._class_means[c_id], dtype=torch.float64) * (
                    0.9 + decay)  # torch.from_numpy(self._class_means[c_id]).to(self._device)
            cls_cov = self._class_covs[c_id]
            # 形成正态分布
            m = MultivariateNormal(cls_mean.float(), cls_cov.float())
            cls_normals[c_id] = m

        sample_list=[64, 62, 60, 59, 57, 55, 54, 52, 51, 49, 48, 46, 45, 44, 42, 41, 40, 39, 37,
 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 26, 25, 24, 23, 23, 22, 21, 21,
 20, 19, 19, 18, 18, 17, 17, 16, 16, 15, 15, 14, 14, 13, 13, 12, 12, 12, 11,
 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

        sample_batch = 16
        for epoch in range(run_epochs):
            losses = 0.
            sampled_data = []
            sampled_label = []

            for c_id in range(crct_num):
                # 采样特征
                samp_num = sample_list[100/crct_num*c_id]
                sampled_data_single = cls_normals[c_id].sample(sample_shape=(samp_num,))
                sampled_data.append(sampled_data_single)  # sampled_data_single：（8,768）
                sampled_label.extend([c_id] * samp_num)
            # 使用采样特征再训练classifier
            sampled_data = torch.cat(sampled_data, dim=0).to(torch.float32)  # （160,768）
            sampled_label = torch.tensor(sampled_label).long()
            inputs = sampled_data
            targets = sampled_label
            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for _iter in range(inputs.size(0) // sample_batch):
                inp = inputs[_iter * sample_batch:(_iter + 1) * sample_batch].cuda()  # 64,768
                tgt = targets[_iter * sample_batch:(_iter + 1) * sample_batch].cuda()
                # 有prototype而没有prompt，则第二阶段tune的是adapter本身——这个分支废弃，因为无法在第二阶段只tune adapter
                if 'prompt' not in self.model_type and 'prototype' in self.model_type:
                    logits = self.custom_clip(inp, image_is_feature=True,test = True)
                else:
                    # -stage two only use classifiers
                    logits= self.custom_clip(inp, image_is_feature=True,test = True)

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
                del inp, tgt
                gc.collect()
            del sampled_data, sampled_label, inputs, targets
            gc.collect()
            scheduler.step()
            # test_acc = self._compute_accuracy(self.model, self.test_loader)
            print('CA Task {} => Loss {:.3f}'.format(
                self.task_id, losses / self._total_classes))
        del cls_normals
        gc.collect()
        logging.info("stage2 tune prompt finishied")

    def offline_evaluate(self, test_loader, classes_names):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label, pred_list = [], []

        text_tokens = self.custom_clip.labels_tokenize(classes_names)
        self.custom_clip.eval()
        with torch.no_grad():
            for data in tqdm(test_loader):
                x, y = data

                x = x.to(self.device)
                y = y.to(self.device)

                logit, _, _ = self.custom_clip(x, text_tokens)
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
                    j = torch.randint(0, self.seen, (1,)).item()
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

    def reset_opt(self, model):
        self.optimizer = select_optimizer(self.opt_name, self.lr, model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer,
                                          None)

    def add_new_batch_class(self, class_name):
        self.batch_exposed_classes = class_name.unique().numpy().tolist()
        # for label in class_name:
        #     if label.item() not in self.batch_exposed_classes:
        #         self.batch_exposed_classes.append(label.item())
        self.batch_exposed_classes_names = [
            self.train_dataset.classes_names[i]
            for i in self.batch_exposed_classes
        ]
        # print(f"**{self.batch_exposed_classes_names}")

    def add_new_class(self, class_name):
        _old_num = len(self.exposed_classes)
        super().add_new_class(class_name)  # 将这批数据的新class加入到exposed_class变量中

        # self.batch_exposed_classes = []
        # self.batch_exposed_classes_names = []
        if self.memory_size > 0:  # 0
            self.batch_exposed_classes = self.exposed_classes
            self.batch_exposed_classes_names = self.exposed_classes_names
        else:
            self.add_new_batch_class(class_name)  # 再将这批数据的新class从exposed_class变量中取出，加入到batch_exposed_classes变量中

    def report_training(self, epoch, sample_num, train_loss, train_acc):
        logging.info(
            f"Train | epoch:{epoch}, Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"Num_Batch_Classes {len(self.batch_exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples - sample_num) / sample_num))}"
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
