import copy
import gc
from typing import Union, List
import numpy as np
import torch
import torch.nn as nn
from .adapter_clip import AdapterCLIP
from .clip import clip_loader
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from .clip.tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class CUSTOM_CLIP(AdapterCLIP):
    def __init__(self, args, device):
        self.classnames = None
        self.name_lens = None
        self.n_class = None
        self.args = args
        super(CUSTOM_CLIP, self).__init__(model_name=args.model_name, device=device,
                                          peft_method='adapter',
                                          peft_encoder=args.peft_encoder)
        clip_model = self.model
        self.dtype = clip_model.dtype
        # self.device = device
        self.logit_scale = clip_model.logit_scale
        self.token_embedding = clip_model.token_embedding
        self.model_type = args.model_type
        self.feature_dim = args.feature_dim
        self.num_prompt = args.num_prompt
        self.n_ctx = args.n_ctx
        self.prompt_prefix = ' '.join(['x'] * self.n_ctx * self.args.topK)
        self.prom_ctx_dim = clip_model.ln_final.weight.shape[0]  # =feature_dim
        if 'prompt' in self.args.model_type:
            # ctx_dim = clip_model.ln_final.weight.shape[0]
            ctx_dim = self.feature_dim
            # init_temp = 'a photo of a '
            # init_emb = self.text_encoder(self.tokenize(init_temp),None,True)
            text_key = torch.empty(self.num_prompt, ctx_dim, dtype=self.dtype)
            nn.init.normal_(text_key, std=0.02)
            text_prompt = torch.empty(self.num_prompt, self.n_ctx, ctx_dim, dtype=self.dtype)
            nn.init.normal_(text_prompt, std=0.02)
            text_key = nn.Parameter(text_key, requires_grad=True)
            text_prompt = nn.Parameter(text_prompt, requires_grad=True)
        else:
            # text端输入为text template
            text_key = None
            text_prompt = None

        # 1. module 1：text prompt encoder
        self.text_encoder = TextEncoder(clip_model)
        if 'prompt' in self.args.model_type:
            # 3. module 3：prompt learner
            self.text_key = text_key
            self.text_prompt = text_prompt
            self.prompt_learner = PromptLearner(self.args, clip_model, self.text_prompt, n_ctx=self.n_ctx)
        # 4. module 4：image encoder

        self.image_encoder = clip_model.visual
        self.dtype = self.image_encoder.conv1.weight.dtype

    def forward(self, image, labels=None, test_class=None, test=False, image_is_feature=False):
        # 1. vision feature入参分为图片和采样特征两种
        if image_is_feature:
            image_features = image
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        else:
            image_features = self.image_encoder(image.type(self.dtype))  # image:(32,3,32,32)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # 使用 .detach() 会返回一个新的张量，这个张量与原始张量共享数据，但不会参与梯度计算。调用 .item() 或 .numpy() 会从张量中提取数据，这些数据不再与计算图关联。
        batch = image_features.shape[0]
        if 'prompt' in self.args.model_type:  # image encoder中有adapter，text端输入为text prompt
            # 2. soft prompt 拼装
            # 2.1 每个 image feature 匹配 2 个prompt
            probability = image_features @ self.text_key.t()
            _, indices = probability.topk(k=min(self.args.topK, probability.shape[1]), dim=1, largest=True)
            key_choose = self.text_key[indices]
            # 2.2 每个image feature 从对应 class 的 attribute cluster 中选择 3 个 attribute
            attr_chose_emb, attr_templa, attr_choose = self.get_img_attrs(image_features, labels, test)  # (32,3,768)
            selected_prompt = self.text_prompt[indices].view(batch, self.n_ctx * self.args.topK, self.feature_dim)
            # 获取 每个 img 得到的 attribute 文本
            attr_ensemble = [",".join(i) for i in attr_choose]
            # 得到其 token 的长度
            ensemble_len = [len(_tokenizer.encode(i)) for i in attr_ensemble]
            tokenized_ensemble = torch.cat([self.tokenize(p) for p in attr_ensemble])  # (n_cls, n_tkn)
            with torch.no_grad():
                ensemble_embedding = self.token_embedding(tokenized_ensemble).type(self.dtype)
            # prompts = []
            # for i in range(self.n_class):
            #     name_len = self.name_lens[i]
            #     prefix_i = self.token_prefix[i:i + 1, :, :].unsqueeze(1)
            #     class_i = self.token_suffix[i:i + 1, :name_len, :].unsqueeze(1)
            #     attrs_i = ensemble_embedding.unsqueeze(0)
            #     suffix_i = self.token_suffix[i:i + 1, name_len:, :].unsqueeze(1)
            #     ctx_i = selected_prompt.unsqueeze(0)
            #     prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=2)
            #     prompts.append(prompt)
            # prompts = torch.cat(prompts, dim=0)
            text_prompt, tokenized_prompts, _, _ = self.prompt_learner(selected_prompt, test, ensemble_embedding)

            text_features = self.text_encoder(text=text_prompt, tokenized_prompts=tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 计算 texual template feature 用于正则化 loss
            template_fea = self.text_encoder(text=attr_templa, tokenized_prompts=None, need_token=True)
            template_fea = template_fea / template_fea.norm(dim=-1, keepdim=True)
            reg_logits = text_features @ template_fea

            # 计算 logits 用于 CE 损失
            text_features = text_features.view(image_features.shape[0], self.n_class, -1)
            image_features = image_features.unsqueeze(1)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * (image_features * text_features).sum(-1)
        else:
            with torch.no_grad():
                text_token = self.labels_tokenize([self.args.text_template.format(c) for c in self.train_cls_name])
                text_features = self.text_encoder(text=text_token, need_token=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * (image_features @ text_features.T)  # logits_per_image

        return logits, None, None

    def get_img_attrs(self, image_features, labels, test, loss=1, cluster=True):

        embed_choose = []
        attr_templa = []
        attr_choose = []
        # tmp = 0 if test else self.args.class_per_task * self.args.sess
        # 3. 每个image feature与对应class的Attribute embed做匹配
        for i in range(image_features.shape[0]):
            # 3.1 取出feature和label，
            ima_fea = image_features[i:i + 1, :]
            # lab = labels[i].item() + tmp
            lab = labels[i].item()  # 传入的 label 就是原本 label 值
            # 3.2 取出该class 的attr str和attr embed
            if cluster:
                embs = []
                attr_strs = []
                for j, cluster in enumerate(self.cluster_info[2][lab]):
                    probability_c = ima_fea @ torch.stack(cluster, dim=0).t()
                    _, ind = torch.max(probability_c, dim=1)  # 取出相似度最高的一个
                    embs.append(cluster[ind.item()])
                    attr_strs.append(self.cluster_info[0][lab][j][ind.item()])
                # todo 这里是把三个 attribute 放一块儿了，下面同理,这与 ArGue 不同，它的 attribute 是分开处理的，用于正则化
                entity_choose = torch.stack(embs, dim=0)
            else:
                ent_str = self.cls_en_map[0][lab]
                ent_embed = self.cls_en_map[1][lab]  # (attriNum,768)
                # 3.3 feature与attr embed做匹配，选出3个attribute 后
                probability_e = ima_fea @ ent_embed.t()  # (1,768)  (attriNum,768)
                _, indices_e = probability_e.topk(k=min(self.args.text_prompt, probability_e.shape[1]), dim=1,
                                                  largest=True)  # (32,3)
                # 3.4 记录匹配的attr str和attr embed
                entity_choose = ent_embed[indices_e]  # indices：（32,3） (32,3,768)
                attr_strs = []
                for j in indices_e.squeeze(0):
                    attr_strs.append(self.cls_en_map[0][lab][j.item()])

            # 3.5 组装texual template
            tempaltes = [f"A photo of a {label}," + ",".join(attr_strs) for label in self.classnames]
            embed_choose.append(entity_choose)
            attr_choose.append(attr_strs)
            attr_templa.append(tempaltes)

        attr_chose_emb = torch.stack(embed_choose, dim=0)
        # 3 个变量分别为img 选择的 attribute feature，选择的 attribute 组装的 template， 选择的 attribute string
        return attr_chose_emb, attr_templa, attr_choose

    def update_class_names(self, new_class_names):
        _num = 0
        for c in new_class_names:
            if c not in self.current_class_names:  # current_class_names在methods的proto_clip里面存着
                self.current_class_names.append(c)
                _num += 1
        if _num > 0:
            # self.text_tokens = self.set_prompt_token(self.current_class_names)#这里set已经没有意义了，train和test都在其他地方做了set
            self.text_tokens = None
        return self.text_tokens

    def set_prompt_token_by_clsname(self, classnames):
        del self.prompt_learner.tokenized_prompts, self.prompt_learner.token_prefix, self.prompt_learner.token_suffix
        self.n_class = len(classnames)
        self.classnames = classnames
        self.name_lens = [len(_tokenizer.encode(name)) for name in self.classnames]
        prompts = [self.prompt_learner.prompt_prefix + ' ' + name + '.' for name in classnames]

        tokenized_prompts = self.tokenize(prompts).cuda()
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts).type(self.dtype).cuda()
        # self.prompt_learner.register_buffer('tokenized_prompts', tokenized_prompts)  # SOS, [n_cls, 1, ctx_dim]
        self.prompt_learner.register_buffer('tokenized_prompts', tokenized_prompts)  # SOS, [n_cls, 1, ctx_dim]
        self.prompt_learner.register_buffer('token_prefix', embedding[:, :1, :])  # SOS, [n_cls, 1, ctx_dim]
        self.prompt_learner.register_buffer('token_suffix', embedding[:, 1 + (self.n_ctx * self.args.topK):,
                                                            :])  # CLS, EOS, [n_cls, -1, ctx_dim]
        self.prompt_learner.n_cls = len(classnames)
        return [self.prompt_learner.token_prefix, self.prompt_learner.token_suffix]

    def extract_vector(self, image):
        image_features = self.image_encoder(image.type(self.dtype))  # image:(32,3,32,32)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.detach()
        return image_features


class PromptLearner(nn.Module):
    def __init__(self, args, clip_model, text_prompt, n_ctx=12, prompt_pos=2):
        super().__init__()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding
        self.args = args
        self.prompt_pos = prompt_pos
        self.dtype = dtype
        self.text_prompt = text_prompt
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.test_prompt_fix = False
        self.tokenized_prompts = None
        self.n_cls = 0
        self.prompt_prefix = ' '.join(['x'] * self.n_ctx * self.args.topK)
        self.token_prefix = None
        self.token_suffix = None

    def forward(self, ctx, infer=False, ensemble_embedding=None):  # 已审，代码没问题
        batch = ctx.shape[0]
        # ctx=self.text_prompt[indices].view(batch, self.n_ctx*self.args.topK, self.ctx_dim)#self.text_prompt[indices]=(batch,3,12,768)->(batch,36,768)
        tokenized_prompts = self.tokenized_prompts.view(self.n_cls, -1)
        n_cls = self.n_cls

        if self.prompt_pos == 2:
            prefix = self.token_prefix.unsqueeze(0).repeat(batch, 1, 1, 1)
            suffix = self.token_suffix.unsqueeze(0).repeat(batch, 1, 1, 1)
            ctx = ctx.unsqueeze(1).repeat(1, n_cls, 1, 1)
            ensemble_embedding = ensemble_embedding.unsqueeze(1).repeat(1, n_cls, 1, 1)#ToDo 会自动截断，观察长度情况
            prompts = torch.cat([prefix, ctx, ensemble_embedding, suffix], dim=2)#todo 这里是 attribute 在前 class在后了，

        # 维度2的尺寸不是1，所以squeeze函数不会起效
        prompts = prompts.squeeze(2).view(batch * self.n_cls, -1,
                                          self.ctx_dim)  # (batch,cls,77,768)->(batch*cls,77,768)
        tokenized_prompts = tokenized_prompts.unsqueeze(0).repeat(batch, 1, 1).view(batch * self.n_cls, -1)
        # self.prompts = prompts
        # self.prompts_token = tokenized_prompts
        if infer:
            return prompts, tokenized_prompts, None, None
        else:
            # nc_prompts, nc_tokenized_prompts = self.only_prefix()
            return prompts, tokenized_prompts, None, None

    def only_prefix(self):
        ctx = self.text_prompt
        prompt_size = ctx.shape[0]
        nc_tokenized_prompts = self.nc_tokenized_prompts.repeat(prompt_size, 1)
        prefix = self.nc_token_prefix.repeat(prompt_size, 1, 1)
        suffix = self.nc_token_suffix.repeat(prompt_size, 1, 1)
        nc_prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return nc_prompts, nc_tokenized_prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text, tokenized_prompts=None, need_token=False):
        # need_token为true，代表已tokenize过，所以需要token_embedding，argmax使用text变量；为false，代表已tokenize+token_embedding过，argmax要使用tokenized_prompts
        x = self.token_embedding(text).type(self.dtype) if need_token else text
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), (text if need_token else tokenized_prompts).argmax(
            dim=-1)] @ self.text_projection
        return x
