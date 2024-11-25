from collections import OrderedDict
from typing import Union, List

import torch
import torch.nn as nn

from .adapter_clip import AdapterCLIP
from .clip.tokenizer import SimpleTokenizer as _Tokenizer
from .clip.zoo import CoPLPrompt

_tokenizer = _Tokenizer()


class MetaNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MetaNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = nn.ReLU(self.fc1(x))
        x = self.fc2(x)
        return x


class CUSTOM_CLIP(AdapterCLIP):
    def __init__(self, args, device):
        self.classnames = None
        self.name_lens = None
        self.n_class = None
        self.args = args
        super(CUSTOM_CLIP, self).__init__(model_name=args.model_name, device=device,
                                          # peft_method='adapter',
                                          peft_method='prefix_prompt',
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
        # self.meta_net = MetaNet(512, (512 // 16), 512)
        prompt_dim = 768
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(prompt_dim, prompt_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(prompt_dim // 16, prompt_dim))]))
        self.prompt_module = CoPLPrompt(768, 10, [100, 8, 0.0])

    def forward(self, image, labels=None, test_class=None, train=True, image_is_feature=False):#image(batch,3,224,224)

        # 1. vision feature入参分为图片和采样特征两种
        if image_is_feature:
            image_features = image
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        else:
            # 1.获取 patch 经 image encoder 之后的 patch feature
            with torch.no_grad():
                patch_features = self.image_encoder.get_patch_feature(image)#batch,196,768
            # 2.获取 patch conditional token
            patch_tokens = self.meta_net(patch_features)#batch,196,768
            # 3.将 patch_token传入，与 prompt 做对齐和加权，再拼接到 attention 的 k v 上，得到最终 image feature
            image_features = self.image_encoder(x=image, prompt_module=self.prompt_module, q=patch_tokens, train=train,
                                                task_id=None)
            # 使用 .detach() 会返回一个新的张量，这个张量与原始张量共享数据，但不会参与梯度计算。调用 .item() 或 .numpy() 会从张量中提取数据，这些数据不再与计算图关联。
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        batch = image_features.shape[0]
        if train:
            if 'prompt' in self.args.model_type:  # image encoder中有adapter，text端输入为text prompt
                # 2. soft prompt 拼装
                # 2.1 每个 image feature 匹配 2 个prompt
                probability = image_features @ self.text_key.t()
                _, indices = probability.topk(k=min(self.args.topK, probability.shape[1]), dim=1, largest=True)
                selected_key = self.text_key[indices]
                selected_prompt = self.text_prompt[indices].view(batch, self.n_ctx * self.args.topK, self.feature_dim)

                # 2.2 每个image feature 从对应 class 的 attribute cluster 中选择 3 个 attribute
                text_prompt, tokenized_prompts, _, _ = self.prompt_learner(selected_prompt, train)
                text_features = self.text_encoder(text=text_prompt, tokenized_prompts=tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logit_scale = self.logit_scale.exp()
                # 计算 logits 用于 CE 损失
                image_features = image_features.unsqueeze(1)
                text_features = text_features.view(image_features.shape[0], self.n_class, -1)
                logits = logit_scale * (image_features * text_features).sum(-1)
            else:
                with torch.no_grad():
                    text_token = self.labels_tokenize([self.args.text_template.format(c) for c in self.train_cls_name])
                    text_features = self.text_encoder(text=text_token, need_token=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                logits = logit_scale * (image_features @ text_features.T)  # logits_per_image
            return logits, image_features, text_features, selected_key
        else:
            probability = image_features @ self.text_key.t()
            _, indices = probability.topk(k=min(self.args.topK, probability.shape[1]), dim=1, largest=True)
            selected_prompt = self.text_prompt[indices].view(batch, self.n_ctx * self.args.topK, self.feature_dim)

            text_prompt, tokenized_prompts, _, _ = self.prompt_learner(selected_prompt, train)
            text_features = self.text_encoder(text=text_prompt, tokenized_prompts=tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            image_features = image_features.unsqueeze(1)
            text_features = text_features.view(image_features.shape[0], self.n_class, -1)
            logits = logit_scale * (image_features * text_features).sum(-1)
            return logits

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
        # self.classnames = classnames
        # self.name_lens = [len(_tokenizer.encode(name)) for name in self.classnames]
        # self.prompt_learner.name_lens = self.name_lens

        prompts = [self.prompt_learner.prompt_prefix + ' ' + name + '.' for name in classnames]
        tokenized_prompts = self.tokenize(prompts).cuda()
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts).type(self.dtype)
        self.prompt_learner.register_buffer('tokenized_prompts', tokenized_prompts)  # SOS, [n_cls, 1, ctx_dim]
        self.prompt_learner.register_buffer('token_prefix', embedding[:, :1, :])  # SOS, [n_cls, 1, ctx_dim]
        self.prompt_learner.register_buffer('token_suffix', embedding[:, 1 + (self.n_ctx * self.args.topK):,
                                                            :])  # CLS, EOS, [n_cls, -1, ctx_dim]
        self.prompt_learner.n_cls = len(classnames)

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

    def forward(self, ctx, train=True):  # 已审，代码没问题
        batch = ctx.shape[0]
        # ctx=self.text_prompt[indices].view(batch, self.n_ctx*self.args.topK, self.ctx_dim)#self.text_prompt[indices]=(batch,3,12,768)->(batch,36,768)
        tokenized_prompts = self.tokenized_prompts.view(self.n_cls, -1)
        n_cls = self.n_cls

        if self.prompt_pos == 2:
            prefix = self.token_prefix.unsqueeze(0).repeat(batch, 1, 1, 1)
            suffix = self.token_suffix.unsqueeze(0).repeat(batch, 1, 1, 1)
            ctx = ctx.unsqueeze(1).repeat(1, n_cls, 1, 1)
            prompts = torch.cat([prefix, ctx, suffix], dim=2)

        # 维度2的尺寸不是1，所以squeeze函数不会起效
        prompts = prompts.squeeze(2).view(batch * self.n_cls, -1,
                                          self.ctx_dim)  # (batch,cls,77,768)->(batch*cls,77,768)
        tokenized_prompts = tokenized_prompts.unsqueeze(0).repeat(batch, 1, 1).view(batch * self.n_cls, -1)
        # self.prompts = prompts
        # self.prompts_token = tokenized_prompts
        if train:
            # nc_prompts, nc_tokenized_prompts = self.only_prefix()
            return prompts, tokenized_prompts, None, None
        else:
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
