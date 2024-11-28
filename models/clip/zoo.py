import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import copy
import math


# Our method!
class CoPLPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)
            # 拼接权重矩阵 W_a
            # W_a = torch.nn.Linear(emb_d * 2, 1)
            # with torch.no_grad():
            #     nn.init.xavier_normal_(W_a.weight)
            #     nn.init.zeros_(W_a.bias)
            # setattr(self, f'e_w_{e}', W_a)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0, 1, 2, 3, 4,5,6]

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]

    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self, f'e_k_{e}')
            A = getattr(self, f'e_a_{e}')
            P = getattr(self, f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)



    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        # 这里的 x_querry就是 patch_token
        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True

            K = getattr(self, f'e_k_{l}')
            A = getattr(self, f'e_a_{l}')
            p = getattr(self, f'e_p_{l}')
            # W_a = getattr(self, f'e_w_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)

            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]


            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            # a_querry = torch.einsum('bd,kd->bkd', x_querry[:,0,:], A)#batch,768 , 10,768 ->batch,10,768
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)#batch,768 , 10,768 ->batch,10,768
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)#batch,10,768  10,768 ->batch,10
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)#batch,10  10,8,768  ->batch,8,768

            # P_ = self.align_patch_and_prompt( prompt_vectors=P_, patch_tokens=x_querry[:,1:,:],W_a=W_a)

            # select prompts
            i = int(self.e_p_length / 2)
            Ek = P_[:, :i, :]
            Ev = P_[:, i:, :]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block

    def align_patch_and_prompt(self, prompt_keys=None, prompt_vectors=None, patch_tokens=None, feature_dim=768,
                               W_a=None):
        '''
        将 prompt 与 patch conditional token 一一对齐，再加权，得到最终的上下文 prompt
        :param prompt:
        :param patch:
        :return:
        '''
        batch_size = patch_tokens.shape[0]  #batch,196,768
        num_patches = patch_tokens.shape[1]
        # num_prompts = prompt_keys.shape[0]
        prompt_length = prompt_vectors.shape[1]  # batch,8,768
        #每个 task 10 个 class，刚好增量训练 10 个 prompt，每个 class 的图片直接用对应的 prompt，让 patch 与该 prompt 的 token 做加权，test 时
        # 1. Attention 权重计算 (公式 3)
        # 扩展维度以便拼接
        patch_expanded = patch_tokens.unsqueeze(2)  # (batch, patch_num, 1, patch_dim)
        prompt_expanded = prompt_vectors.unsqueeze(1)  # (batch, 1, prompt_length, prompt_dim)

        # 拼接 patch tokens 和 prompt tokens
        sp_concat = torch.cat([patch_expanded.repeat(1, 1, prompt_length, 1),
                               prompt_expanded.repeat(1, num_patches, 1, 1)], dim=-1)
        # 拼接后 sp_concat 的形状为 (batch, patch_num, prompt_length, patch_dim + prompt_dim)

        # 使用 W_a 计算 score 并通过 tanh 激活
        scores = torch.tanh(W_a(sp_concat)).squeeze(-1)  # (batch, patch_num, prompt_length)prompt_length=tokens

        # Softmax 归一化得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # (batch, patch_num, prompt_length)

        # 2. 上下文向量生成 (公式 5)
        # 使用 einsum 进行加权求和，得到每个 prompt token 的上下文表示   todo 把新的训练参数加进去
        context_vectors_per_patch = torch.einsum('bpl,bld->bpld', attention_weights, prompt_vectors)
        # context_vectors_per_patch 的形状为 (batch, prompt_length, prompt_dim)

        # 3. 更新 prompt tokens (公式 6)
        # 对所有 patch 的上下文向量加和
        context_vectors = prompt_vectors+ context_vectors_per_patch.sum(dim=1)  # (batch, prompt_length, prompt_dim)

        # # 拼接 (batch, patches, prompts, feature_dim * 2)
        # sp_vi_concat = torch.cat([patch_tokens, prompt_vectors], dim=1)
        #
        # # 计算注意力分数
        # scores = torch.tanh(W_a(sp_vi_concat)).squeeze(-1)  # (batch, patches, prompts)
        #
        # # 归一化注意力分数
        # attention_weights = torch.softmax(scores, dim=-1)  # (batch, patches, prompts)
        #
        # # 公式 (5)：加权求和生成上下文向量
        # context_vectors = torch.einsum('bpn,nf->bpf', attention_weights, prompt_vectors)
        #
        # print("Context Vectors Shape:", context_vectors.shape)  # (batch, patches, feature_dim)
        #
        return context_vectors


    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda()) ** 2).mean()





# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p
