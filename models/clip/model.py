from typing import Tuple, Union
from collections import OrderedDict, Counter

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from .sparse_dispatcher import SparseDispatcher
from .adapter import Adapter
from .lora import MultiheadAttention as LoRAMultiheadAttention


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict([("-1", nn.AvgPool2d(stride)),
                             ("0",
                              nn.Conv2d(inplanes,
                                        planes * self.expansion,
                                        1,
                                        stride=1,
                                        bias=False)),
                             ("1", nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):

    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1],
                      x.shape[2] * x.shape[3]).permute(2, 0,
                                                       1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self,
                 layers,
                 output_dim,
                 heads,
                 input_resolution=224,
                 width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3,
                               width // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2,
                               width // 2,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2,
                               width,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim,
                                        heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2),
                             (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)),
                         ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype,
            device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False,
                         attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_prefix(ResidualAttentionBlock):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None):
        super().__init__(d_model, n_head, attn_mask)

        # 将 attention 模块替换成加入 prompt 的
        # self.attn = PreT_Attention(d_model, n_head)
        self.attn = Attention_CODA(d_model, n_head)

    def forward(self, x: torch.Tensor, register_hook=False, prompt=None):
        x = x + self.attn(x=self.ln_1(x), register_hook=register_hook,prompt=prompt)
        # x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_LoRA(ResidualAttentionBlock):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 design_details: dict = {}):
        super().__init__(d_model, n_head, attn_mask)

        self.lora_alpha = design_details.get('lora_alpha', 1)
        self.lora_r = design_details.get('lora_r', 4)

        self.attn = LoRAMultiheadAttention(d_model,
                                           n_head,
                                           lora_alpha=self.lora_alpha,
                                           r=self.lora_r)


class ResidualAttentionBlock_Adapter(ResidualAttentionBlock):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 design_details: dict = {}):
        super().__init__(d_model, n_head, attn_mask)

        self.ffn_num = design_details.get('ffn_num', 64)

        # adapter
        self.adaptmlp = Adapter(
            d_model=d_model,
            dropout=0.1,
            bottleneck=self.ffn_num,
            init_option='lora',
            adapter_scalar=0.1,
            adapter_layernorm_option='none',
        )

    def forward(self, x: torch.Tensor):
        x = x + self.adaptmlp(self.attention(self.ln_1(x.clone())))
        x = x + self.adaptmlp(self.mlp(self.ln_2(x.clone())))
        return x


class ResidualAttentionBlock_MoA(ResidualAttentionBlock):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 modal=None,
                 design_details: dict = {}):
        super().__init__(d_model, n_head, attn_mask)

        self.top_k = design_details.get('top_k', 2)
        self.ffn_num = design_details.get('ffn_num', 64)
        self.experts_num = design_details.get('experts_num', 2)
        self.noisy_gating = design_details.get('noisy_gating', True)

        self.softmax = nn.Softmax(1)
        self.softplus = nn.Softplus()
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.modal = modal
        if modal == 'text':
            self.choose_map_text = torch.zeros([self.experts_num])
        else:
            self.choose_map_image = torch.zeros([self.experts_num])

        # router
        self.router = nn.Parameter(torch.zeros(d_model, self.experts_num),
                                   requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d_model, self.experts_num),
                                    requires_grad=True)

        # adapter
        self.adaptmlp_list = nn.ModuleList()
        for i in range(self.experts_num):
            self.adaptmlp = Adapter(
                d_model=d_model,
                dropout=0.1,
                bottleneck=self.ffn_num,
                init_option='lora',
                adapter_scalar=0.1,
                adapter_layernorm_option='none',
            )
            self.adaptmlp_list.append(self.adaptmlp)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev,
                       noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        # print('1231',clean_values)  # 全nan
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(
            batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        #

        prob_if_in = normal.cdf(
            (clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf(
            (clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, w_gate, w_noise, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        clean_logits = x @ w_gate.to(x)
        if self.noisy_gating and self.training:
            raw_noise_stddev = x @ w_noise.to(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) *
                                           noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1,
                                                  self.experts_num),
                                              dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.top_k < self.experts_num and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits,
                                        noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))

        x_re = x.permute(1, 0, 2)[:, 0, :]
        gates, _ = self.noisy_top_k_gating(x_re, self.router, self.w_noise)

        nonzero_indices = torch.nonzero(gates)
        counter = Counter(nonzero_indices[:, 1].tolist())
        for number, count in counter.items():
            if self.modal == 'text':
                self.choose_map_text[
                    number] = self.choose_map_text[number] + count
            else:
                self.choose_map_image[
                    number] = self.choose_map_image[number] + count

        dispatcher = SparseDispatcher(self.experts_num, gates)
        expert_inputs = dispatcher.dispatch(
            x.permute(1, 0, 2).view(x.shape[1], -1))
        expert_outputs = [
            self.adaptmlp_list[i](expert_inputs[i].view(
                expert_inputs[i].shape[0], x.shape[0], x.shape[2]).to(x),
                                  add_residual=False)
            for i in range(self.experts_num)
        ]

        i = 0
        while i < len(expert_outputs):
            if expert_outputs[i].shape[0] == 0:
                expert_outputs.pop(i)
            else:
                expert_outputs[i] = expert_outputs[i].view(
                    expert_outputs[i].shape[0], -1)
                i += 1

        y = dispatcher.combine(expert_outputs)
        y = y.view(x.shape[1], x.shape[0], x.shape[2])

        x = x + self.mlp(self.ln_2(x)) + y.permute(1, 0, 2)

        return x


class Transformer(nn.Module):

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None,
                 design_details: dict = {},
                 modal='text'):
        super().__init__()
        self.width = width
        self.layers = layers

        res_type = design_details.get('method', 'vanilla')
        peft_flag = design_details.get('peft_encoder',
                                       'none') in ['both', modal]

        if res_type == 'moe' and peft_flag:
            self.resblocks = nn.Sequential(*[
                ResidualAttentionBlock_MoA(width, heads, attn_mask, modal,
                                           design_details)
                for _ in range(layers)
            ])
        elif res_type == 'adapter' and peft_flag:
            self.resblocks = nn.Sequential(*[
                ResidualAttentionBlock_Adapter(width, heads, attn_mask,
                                               design_details)
                for _ in range(layers)
            ])
        elif res_type == 'lora' and peft_flag:
            self.resblocks = nn.Sequential(*[
                ResidualAttentionBlock_LoRA(width, heads, attn_mask,
                                            design_details)
                for _ in range(layers)
            ])
        elif res_type == 'prefix_prompt' and peft_flag:  # 只往 visual 端加 prompt
            self.resblocks = nn.Sequential(*[
                ResidualAttentionBlock_prefix(width, heads)
                for _ in range(layers)
            ])
        else:
            self.resblocks = nn.Sequential(*[
                ResidualAttentionBlock(width, heads, attn_mask)
                for _ in range(layers)
            ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)



class PreT_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            # prefix key, value
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous()  # 2, B, num_heads, prompt_length, C // num_heads
            key_prefix = prompt[0]  # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[1]  # B, num_heads, prompt_length, embed_dim // num_heads

            expected_shape = (B, self.num_heads, C // self.num_heads)

            assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[
                3]) == expected_shape, f'key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}'
            assert (value_prefix.shape[0], value_prefix.shape[1], value_prefix.shape[
                3]) == expected_shape, f'value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}'

            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)
            # print(key_prefix.shape)
        # print(q.shape, k.shape, v.shape)
        # print(k.transpose(-2, -1).shape)

        # exit()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print("attn", attn.shape)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print("x", x.shape)
        # exit()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_CODA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, register_hook=False, prompt=None):
        B, N, C = x.shape # B是batch，N是patchNum +1，C就是特征维度，目前x是batch，197,768
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            pk, pv = prompt
            pk = pk.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            pv = pv.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VisualTransformer(nn.Module):

    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 modal=None,
                 design_details: dict = {}):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        # Added so this info is available. should not change anything.
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # self.prompt_module = CoPLPrompt(768, 100, 8)

        self.transformer = Transformer(width,
                                       layers,
                                       heads,
                                       modal=modal,
                                       design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def get_patch_feature(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
            dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)  # input,output:(32,257,1024),position:(257,1024)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD(batch,197,768)

        # for i, blk in enumerate(self.transformer.resblocks):
        #     x = blk(x)


        x = self.ln_post(x)# batch,196,768)
        # x = self.ln_post(x[:, 0, :])# batch,768)

        # if self.proj is not None:
        #     x = x @ self.proj

        return x  #就是下面的 q

    def forward(self, x, prompt_module=None, register_blk=-1, q=None, train=False, task_id=None):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)#batch,196,768
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
            dim=1)  # shape = [*, grid ** 2 + 1, width]  batch,197,768
        x = x + self.positional_embedding.to(x.dtype)  # batch,197,768

        x = self.ln_pre(x)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        prompt_loss = torch.zeros((1,), requires_grad=True).cuda()
        for i, blk in enumerate(self.transformer.resblocks):

            if prompt_module is not None:
                if train:
                    p_list, loss, x = prompt_module.forward(q, i, x, train=True, task_id=task_id)
                    prompt_loss += loss
                else:
                    p_list, _, x = prompt_module.forward(q, i, x, train=False, task_id=task_id)
            else:
                p_list = None

            x = blk(x, register_blk == i, prompt=p_list)

        # x = x.permute(1, 0, 2)  # LND -> NLD(batch,197,768)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            # vision
            image_resolution: int,
            vision_layers: Union[Tuple[int, int, int, int], int],
            vision_width: int,
            vision_patch_size: int,
            # text
            context_length: int,
            vocab_size: int,
            transformer_width: int,
            transformer_heads: int,
            transformer_layers: int,
            design_details: dict):
        super().__init__()
        self.design_details = design_details

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(layers=vision_layers,
                                         output_dim=embed_dim,
                                         heads=vision_heads,
                                         input_resolution=image_resolution,
                                         width=vision_width)
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(input_resolution=image_resolution,
                                            patch_size=vision_patch_size,
                                            width=vision_width,
                                            layers=vision_layers,
                                            heads=vision_heads,
                                            output_dim=embed_dim,
                                            modal='image',
                                            design_details=design_details)

        self.transformer = Transformer(width=transformer_width,
                                       layers=transformer_layers,
                                       heads=transformer_heads,
                                       attn_mask=self.build_attention_mask(),
                                       modal='text',
                                       design_details=design_details)

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        self._text_features = None
        self._image_features = None

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                self.visual.layer1, self.visual.layer2, self.visual.layer3,
                self.visual.layer4
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * (
                (2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection,
                            std=self.transformer.width ** -0.5)

    def load_pretrain(self,state_dict):
        checkpoint= state_dict
        # rename the keys in checkpoint
        clip_weight_dict = {'in_proj_weight': 'qkv.weight',
                            'in_proj_bias': 'qkv.bias',
                            'out_proj.weight': 'proj.weight',
                            'out_proj.bias': 'proj.bias'}

        weight_key = list(checkpoint.keys())

        for k in weight_key:
            for clip_k, new_k in clip_weight_dict.items():
                if clip_k in k:
                    checkpoint[k.replace(clip_k, new_k)] = checkpoint.pop(k)
                    continue

        return checkpoint


    def key_replace(self,state_dict):
        for key in list(state_dict.keys()):
            if 'qkv.weight' in key:
                qkv_weight = state_dict.pop(key)
                q_weight = qkv_weight[:768]
                k_weight = qkv_weight[768:768 * 2]
                v_weight = qkv_weight[768 * 2:]
                state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
                state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
                state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
            elif 'qkv.bias' in key:
                qkv_bias = state_dict.pop(key)
                q_bias = qkv_bias[:768]
                k_bias = qkv_bias[768:768 * 2]
                v_bias = qkv_bias[768 * 2:]
                state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
                state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
                state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
        return state_dict

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):

        x = self.token_embedding(text).type(
            self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]),
        text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        image_features = self.encode_image(image)  # image(32,3,224,224),output:32,768
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        text_features = text_features / text_features.norm(dim=-1,
                                                           keepdim=True)

        # if self.baseline:
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text, image_features, text_features


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias", "bias_k", "bias_v"
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, design_details: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([
            k for k in state_dict.keys()
            if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        ])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2] for k in state_dict
                    if k.startswith(f"visual.layer{b}")))
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] -
             1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict[
            "visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2] for k in state_dict
            if k.startswith(f"transformer.resblocks")))

    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width,
                 vision_patch_size, context_length, vocab_size,
                 transformer_width, transformer_heads, transformer_layers,
                 design_details)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    state_dict = model.load_pretrain(state_dict)#让 attention 模块参数加载进来

    # 这里没有convert_weights方法
    state = model.load_state_dict(state_dict, strict=False)
    # state = model.load_state_dict(state_dict)
    for p in model.parameters():
        p.data = p.data.float()
    return model.eval()




