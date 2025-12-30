import torch.nn.functional as func
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Tuple
from einops import repeat, rearrange

from collections import OrderedDict
from timm.layers import trunc_normal_
import torch.distributed as dist
import torch.autograd as autograd
from torchvision import models

from . import FakeVAE
from .input_adapters import FeaturesInputAdapter
from .register import register_model

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualCrossAttentionBlock(nn.Module):
    """Cross-attention module between 2 inputs. """

    def __init__(self, d_model: int, n_heads: int,
                 add_bias_kv: bool = False,
                 dropout: float = 0.,
                 batch_first: bool = False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_heads, add_bias_kv=add_bias_kv,
                                          dropout=dropout, batch_first=batch_first)
        self.ln_1x = nn.LayerNorm(d_model)
        self.ln_1y = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: torch.Tensor = None,
                  attn_mask: torch.Tensor = None):
        return self.attn(x, y, y, need_weights=False, key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: torch.Tensor = None,
                attn_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1x(x), self.ln_1y(y), key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class ResidualAttentionBlock(nn.Module):
    """Self-attention block"""

    def __init__(self, d_model: int, n_head: int,
                 add_bias_kv: bool = False,
                 dropout: float = 0.,
                 batch_first: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, add_bias_kv=add_bias_kv,
                                          dropout=dropout, batch_first=batch_first)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        return self.attn(x.clone(), x, x, need_weights=False, key_padding_mask=key_padding_mask)[0]

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class FusionTransformer(nn.Module):
    """
    input: 2D tensor (batch_size, seq_len, embedding_dim)
    output: 2D tensor (batch_size, embedding_dim)
    """

    def __init__(self, width: int,
                 n_heads: int,
                 n_layers: int,
                 num_mask: int,
                 mask_ratio: float,
                 fusion: str = "concat",
                 pool: str = "cls",
                 mask_grad: bool = True,
                 random_mask: bool = True,
                 add_bias_kv: bool = False,
                 dropout: float = 0.,
                 batch_first: bool = True):
        """
        :param width: embedding size
        :param n_heads: number of heads in multi-head attention blocks
        :param n_layers: number of attention blocks
        :param num_mask: number of masked view
        :param mask_ratio: ratio of masking
        :param fusion: "concat" or "x-attn"
        :param pool: "cls" or "pool"
        :param mask_grad: if `True`, gradient is computed for masked tokens, otherwise, masked tokens are set to 0
        :param random_mask: if `True`, randomly mask tokens
        :param add_bias_kv: If specified, adds bias to the key and value sequences at dim=0.
        :param dropout: Dropout probability on `attn_output_weights`
        :param batch_first: input tensor is either (batch, tokens, features) if `True` or (tokens, batch, features)
        """
        super().__init__()

        self.fusion = fusion
        self.width = width
        self.layers = n_layers
        self.num_mask = num_mask
        self.mask_ratio = mask_ratio
        self.random_mask = random_mask
        self.mask_grad = mask_grad
        self.norm = nn.LayerNorm(width)
        self.token_dim = 1 if batch_first else 0
        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, width)) if self.pool == "cls" else None
        if fusion == "concat":
            self.resblocks = nn.Sequential(*[
                ResidualAttentionBlock(width, n_heads, add_bias_kv=add_bias_kv,
                                       dropout=dropout, batch_first=batch_first)
                for _ in range(n_layers)])
        elif fusion == "x-attn":
            self.resblocks = [
                nn.Sequential(*[
                    ResidualCrossAttentionBlock(width, n_heads, add_bias_kv=add_bias_kv,
                                                dropout=dropout, batch_first=batch_first)
                    for _ in range(n_layers)])
                for _ in range(2)]
        else:
            raise ValueError("Unknown fusion %s" % fusion)
        self.initialize()

        if mask_grad:
            self.mask_tokens = nn.Parameter(torch.zeros(1, 1, width))
            self._trunc_normal_(self.mask_tokens, std=.02)
        else:
            self.mask_tokens = torch.zeros(1, 1, width).cuda()

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def initialize(self):
        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def random_masking(self, x, ratio):
        """
        Randomly mask tokens in the input tensor.
        """
        if self.token_dim == 1:
            N, L, E = x.shape
            dim_feature = 2
        else:  # token_dim == 2
            N, E, L = x.shape
            dim_feature = 1

        len_keep = int(L * (1 - ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        ids_shuffle = torch.argsort(noise, dim=1)
        # 最终还是找前几个最小的
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.zeros([N, L], device=x.device)
        mask[:, :len_keep] = 1

        mask = torch.gather(mask, dim=1, index=ids_restore).unsqueeze(dim_feature)
        return mask  # [N, L, 1]

    def forward(self, x: List[torch.Tensor], key_padding_mask: List[torch.Tensor] = None):
        """
        :param x: input tensors
        :param key_padding_mask: torch mask of type bool. `True` indicates unattended tokens.
        :return:
        """
        # Concatenate over tokens + self-attention
        if self.fusion == "concat":
            # compute masked representation
            mask_outputs = []
            if self.random_mask:
                mask_outputs = [self.get_mask_no_cross(x, key_padding_mask) for i in range(self.num_mask)]

            # compute multimodal representation
            x = torch.cat(x, dim=self.token_dim)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(key_padding_mask, dim=self.token_dim)
            if self.pool == "cls":  # append cls token at the beginning
                cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
                x = torch.cat((cls_token, x), dim=self.token_dim)
                if key_padding_mask is not None:
                    key_padding_mask = torch.cat(
                        (torch.zeros_like(cls_token[:, :, 0]), key_padding_mask), dim=self.token_dim)

            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.masked_fill(key_padding_mask.bool(), float("-inf")).float()

            for layer in self.resblocks:
                x = layer(x, key_padding_mask=key_padding_mask)

            x = self.norm(x)

            if self.pool == "cls":
                x = x[:, 0] if self.token_dim == 1 else x[0]
            else:
                x = x.mean(dim=self.token_dim)
            return x, mask_outputs

        # Cross-attention + concatenate over tokens
        elif self.fusion == "x-attn":
            if self.pool == "cls":
                raise ValueError("Only `mean` pool is implemented for cross-attention.")
            if len(x) != 2:
                raise ValueError("Only 2 modalities are currently accepted for cross-attention")
            if key_padding_mask is not None:
                raise NotImplementedError()
            x1, x2 = x
            x = torch.cat([self.resblocks[0](x1, x2, key_padding_mask),
                           self.resblocks[1](x2, x1, key_padding_mask)], dim=self.token_dim)
            x = self.norm(x).mean(dim=self.token_dim)
            return x

    def get_mask_no_cross(self, x: List[torch.Tensor], key_padding_mask: List[torch.Tensor] = None):
        mask = torch.cat([self.random_masking(x[i], self.mask_ratio) for i in range(len(x))],
                         dim=self.token_dim)  # N,L*len,1
        x = torch.cat(x, dim=self.token_dim)  # N,L,E; mask_tokens: 1,1,E
        mask_outputs = x * mask + self.mask_tokens * (1 - mask)  # [N,L,E]

        if key_padding_mask is not None:
            key_padding_mask = torch.cat(key_padding_mask, dim=self.token_dim)
        if self.pool == "cls":  # append cls token at the beginning
            cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=mask_outputs.shape[0])
            mask_outputs = torch.cat((cls_token, mask_outputs), dim=self.token_dim)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    (torch.zeros_like(cls_token[:, :, 0]), key_padding_mask), dim=self.token_dim)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.masked_fill(key_padding_mask.bool(), float("-inf")).float()

        # mask fusion
        for layer in self.resblocks:
            mask_outputs = layer(mask_outputs, key_padding_mask=key_padding_mask)
        mask_outputs = self.norm(mask_outputs)
        if self.pool == "cls":
            mask_outputs = mask_outputs[:, 0] if self.token_dim == 1 else mask_outputs[0]
        else:
            mask_outputs = mask_outputs.mean(dim=self.token_dim)
        return mask_outputs  # [N, E]


class InfFusion(nn.Module):
    def __init__(self,
                 encoders: List[nn.Module],
                 input_adapters: List[nn.Module],
                 mask_ratio: float = 0.5,
                 num_mask: int = 5,
                 embed_dim: int = 512,
                 fusion: str = "concat",
                 pool: str = "cls",
                 n_heads: int = 8,
                 n_layers: int = 1,
                 mask_grad: bool = True,
                 random_mask: bool = True,
                 add_bias_kv: bool = False,
                 dropout: float = 0.):
        """ Multi-Modal (MM) fusion model using `FusionTransformer` in the latent space.
        It can handle an arbitrary number of input modalities.
        Each modality is encoded through either a:
            - Transformer (e.g. for text or audio) -> no adapters
            - CNN (e.g. for images) -> `PatchedInputAdapter` for tokenization
            - MLP (e.g. tabular data) -> `FeaturesInputAdapter` for tokenization
        Once each modality is encoded and tokenized, it then goes to `FusionTransformer` to output
        the final embedding.

        :param encoders: List of Torch encoders (CNN, Transformer, MLP, etc.) for each modality
        :param input_adapters: List of Torch adapters for each modality (can be None if not required)
        :param mask_ratio: Ratio of masking
        :param num_mask: Number of masked view
        :param embed_dim: Embedding size
        :param fusion: "concat" or "x-attn". For "x-attn", only "mean" pool is accepted.
        :param pool: "cls" or "mean", pooling strategy for the tokens
        :param n_heads: Number of heads in multi-heads attention blocks
        :param n_layers: Number of attention layers in latent fusion
        :param mask_grad: If `True`, gradient is computed for masked tokens, otherwise, masked tokens are set to 0
        :param random_mask: If `True`, randomly mask tokens
        :param add_bias_kv: If `True`, add bias term in key/values mapping
        :param dropout: attention matrix dropout rate
        """
        super().__init__()
        assert len(encoders) == len(input_adapters), "Each encoder must have an adapter."
        assert pool in {'cls', 'mean'}, "pool type must be either cls (cls token) or mean (mean pooling)"
        self.input_adapters = nn.ModuleList(input_adapters)
        self.encoders = nn.ModuleList(encoders)
        self.pool = pool
        self.num_modalities = len(self.encoders)
        self.fusion_transformer = FusionTransformer(embed_dim, n_heads, n_layers, num_mask, mask_ratio,
                                                    fusion, pool, mask_grad, random_mask, add_bias_kv, dropout,
                                                    batch_first=True)

    def forward(self, x: List[torch.Tensor],
                mask_modalities: Optional[Union[List[bool], List[List[bool]]]] = None):
        """
        :param x: List of tensors
        :param mask_modalities: Mask indicating which modalities are given.
            By default, `x` should have all modalities.
            If a list of lists is given, assume `x` has all modalities and computes
            a list of output by masking out modalitites according to `mask_modalities`.
        :return: a latent vector z or list of vector if `mask_modalities` is a list of list.
        """
        list_mask_mod = None
        if mask_modalities is None:
            mask_modalities = self.num_modalities * [True]
        elif isinstance(mask_modalities, list) and len(mask_modalities) > 0 and isinstance(mask_modalities[0], list):
            list_mask_mod = mask_modalities
            mask_modalities = self.num_modalities * [True]

        assert len(mask_modalities) == self.num_modalities, (
            f"Mask size does not match `num_modalities`: {len(mask_modalities)} != {self.num_modalities}")

        num_modalities = sum(mask_modalities)
        assert len(x) == num_modalities, (
            f"Incorrect number of inputs: {len(x)} != {num_modalities}")
        encoders = [enc for (enc, m) in zip(self.encoders, mask_modalities) if m]
        input_adapters = [adapter for (adapter, m) in zip(self.input_adapters, mask_modalities) if m]
        attn_mask = []

        # 1. Encode input modalities
        z, mask_outputs = [], []
        for (enc, xi) in zip(encoders, x):
            embedding = enc(xi)
            attn_mask_ = None
            if isinstance(embedding, dict):  # attention mask must be considered
                attn_mask_ = embedding["attention_mask"]
                embedding = embedding["token_embeddings"]
            z.append(embedding)
            attn_mask.append(attn_mask_)

        # 2. Tokenize each latent features
        latent_tokens = [adapter(zi) if adapter is not None else zi
                         for (adapter, zi) in zip(input_adapters, z)]
        attn_mask = [attn_mask_ if attn_mask_ is not None else torch.zeros_like(zi[:, :, 0]).bool()
                     for (attn_mask_, zi) in zip(attn_mask, latent_tokens)]
        if list_mask_mod is None:
            # 3. FusionTransformer forward pass
            z, mask_outputs = self.fusion_transformer(latent_tokens, key_padding_mask=attn_mask)
        else:
            # 3.bis Drop modalities according to `mask_modalities`
            z, mask_outputs = [], []
            for mask_mod in list_mask_mod:
                latent_tokens_ = [z for (z, m) in zip(latent_tokens, mask_mod) if m]
                attn_mask_ = [attn for (attn, m) in zip(attn_mask, mask_mod) if m]
                # 3. FusionTransformer forward pass
                res1, res2 = self.fusion_transformer(latent_tokens_)
                z.append(res1)
                mask_outputs.append(res2)
        return z, mask_outputs

    def encode_single_mod(self, x: torch.Tensor, mod: int):
        assert 0 <= mod < self.num_modalities, "Wrong input modality"
        return self.encoders[mod](x)

#最大化特征熵来避免模型坍塌
def entropy_gradeint(keys):
   dlog_q=com_score(keys)
   grad_en=torch.mean(torch.sum(-dlog_q.detach()*keys,-1))
   return grad_en

@torch.no_grad()
def com_score(keys, eta=0.01):
    batch_size = keys.size()[0]
    pairwise_similar = torch.mm(keys, torch.t(keys))  # 计算相似度
    tau = heuristic_kernel_width(keys, keys, pairwise_similar)  # 核宽度，当特征高度相似，导致pairwise_similar趋于0，tau过小
    tau = torch.clamp(tau, min=0.1)  # 防止tau过小,当tau的值为0.0787,就会爆炸
    scaled_similar = pairwise_similar / tau
    # max_val = torch.max(scaled_similar)
    # scaled_similar = scaled_similar - max_val  # 避免指数爆炸

    Gram = torch.exp(scaled_similar)  # 核矩阵，会出现指数爆炸，当tau过下的时候
    x_row = torch.unsqueeze(keys, -2)
    diff = x_row / tau
    grad_x = torch.sum(Gram.unsqueeze(-1) * diff, -2)
    Gram_ivs = torch.inverse(Gram + eta * torch.eye(batch_size).cuda())

    dlog_q = -torch.einsum('ik,kj->ij', [Gram_ivs, grad_x])
    return dlog_q


@torch.no_grad()
def heuristic_kernel_width(x_samples, x_basis, pairwise_similar):
    n_samples = x_samples.size()[-2]
    n_basis = x_basis.size()[-2]
    #   pairwise_dist = torch.arccos(pairwise_similar)
    pairwise_dist = 1 - pairwise_similar
    k = n_samples * n_basis // 2
    top_k_values = torch.topk(torch.reshape(pairwise_dist, [-1, n_samples * n_basis]), k=k)[0]
    kernel_width = torch.reshape(top_k_values[:, -1], x_samples.size()[:-2])
    return kernel_width.detach()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()



class GatherLayer(autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]



def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor



class InfMaskingLoss(nn.Module):
    def __init__(self, temperature=0.1, weights=None, cross=False, only_mask_last=False,
                 mask_lambda=0.25, penalty=None):
        super().__init__()
        self.temperature = temperature
        self.weights = weights
        self.mask_lambda = mask_lambda
        self.cross = cross
        self.INF = 1e8
        self.penalty = penalty
        self.only_mask_last = only_mask_last

    def infonce(self, z1, z2):  # InfoNCE Loss
        N = len(z1)
        sim_zii = (z1 @ z1.T) / self.temperature  # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z2 @ z2.T) / self.temperature  # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z1 @ z2.T) / self.temperature  # dim [N, N] => the diag contains the correct pairs (i,j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()
        # compute SSL accuracy
        with torch.no_grad():
            pred = torch.argmax(sim_zij, dim=1)
            correct = pred.eq(torch.arange(N, device=z1.device)).sum()
            acc = 100 * correct / N
        return loss, acc

    def get_mask_loss(self, q, mask_out, ratio):
        """
        :param q: [bsize, feature_dim]
        :param mask_out: List of tensors with shape (bsize, feature_dim), length T
        :param ratio: float
        """
        B, E = q.shape
        T = len(mask_out)
        k_pos = torch.cat(mask_out, dim=0)  # dim [T*B, E]

        # compute mean and covariance
        k_tryies = k_pos.shape[0] / B
        k_tryies = int(k_tryies)
        k_new_shape = k_pos.view(k_tryies, B, -1)  # dim [T, B, E]

        # compute mean
        k_means = torch.mean(k_new_shape, dim=0, keepdim=True)  # dim [1, B, E]
        value_minus_mean = k_new_shape - k_means  # minus mean, dim [T, B, E]
        # compute covariance
        value_minus_mean_part_1 = value_minus_mean.permute(1, 0, 2)  # dim [B, T, E]
        value_minus_mean_part_2 = value_minus_mean.permute(1, 2, 0)  # dim [B, E, T]
        k_sigma = torch.bmm(value_minus_mean_part_2, value_minus_mean_part_1) / k_tryies  # dim [B, E, E]

        # apply softplus to covariance
        k_sigma = k_sigma * ratio / self.temperature

        # positive logits: Bx1
        k_means = k_means.squeeze(dim=0)  # dim [B, E]
        l_pos = torch.einsum('nc,nc->n',
                             [q, k_means + 0.5 * torch.bmm(k_sigma, q.unsqueeze(dim=-1)).squeeze(dim=-1)]).unsqueeze(
            -1)  # dim [B, 1]

        k_neg = torch.stack(mask_out, dim=0)  # dim [T, B, E]

        l_neg = torch.einsum('be,tne->tbn', [q, k_neg])  # dim [T, B, B]
        # delete the diagonal
        l_neg = l_neg - self.INF * torch.eye(B, device=q.device).unsqueeze(0).expand(T, -1, -1)  # dim [T, B, B]
        l_neg = list(torch.unbind(l_neg, dim=0))
        self_sim = torch.einsum('nc,bc->nb', [q, q]) - self.INF * torch.eye(B, device=q.device)
        # add self-similarity to the negative logits
        l_neg.append(self_sim)
        # .append(torch.einsum('nc,bc->nb', [q, q])-self.INF * torch.eye(B, device=q.device)) # dim [T+1, B, B]
        l_neg = torch.cat(l_neg, dim=1)  # dim [B, (T+1)*B]

        # logits: Bx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        # compute the masklabel
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        mask_loss = 0.5 * torch.bmm(torch.bmm(q.reshape(B, 1, E), k_sigma),
                                    q.reshape(B, E, 1)).mean() / self.temperature

        criterion = nn.CrossEntropyLoss().cuda()
        cls_loss = criterion(logits, labels)

        return mask_loss + cls_loss

    def forward(self, outputs):
        """
        :param outputs: Dict
            Dictionary with keys:
                - "aug1_embed", List of tensors with shape (bsize, feature_dim), 1st aug.
                - "aug2_embed", List of tensors with shape (bsize, feature_dim), 2nd aug.
                - "prototype", integer indicating where the multimodal representation Z is stored in "aug1_embed" and "aug2_embed".
                - "mask_out1", List of tensors with shape (bsize, feature_dim), 1st aug.
                - "mask_out2", List of tensors with shape (bsize, feature_dim), 2nd aug.
                - "epoch", current epoch number.
                - "max_epochs", total number of epochs.
        :return: {"loss": torch.Tensor(float), "ssl_acc": torch.Tensor(float)}
        """
        z1, z2, prototype = outputs["aug1_embed"], outputs["aug2_embed"], outputs["prototype"]
        mask_out1, mask_out2 = outputs["mask_out1"][prototype], outputs["mask_out2"][prototype]  # List
        epoch, max_epochs = outputs["epoch"], outputs["max_epochs"]

        assert len(z1) == len(z2)
        assert len(mask_out1) == len(mask_out2)

        n_emb = len(z1)
        z1 = [func.normalize(z, p=2, dim=-1) for z in z1]
        z2 = [func.normalize(z, p=2, dim=-1) for z in z2]
        Z = all_gather_batch_with_grad(z1 + z2)
        z1, z2 = Z[:n_emb], Z[n_emb:]

        mask_out1 = [func.normalize(mask_z, p=2, dim=-1) for mask_z in mask_out1]
        mask_out2 = [func.normalize(mask_z, p=2, dim=-1) for mask_z in mask_out2]
        MASK_OUT = all_gather_batch_with_grad(mask_out1 + mask_out2)
        num_mask = len(mask_out1)
        mask_out1, mask_out2 = MASK_OUT[:num_mask], MASK_OUT[num_mask:]

        loss = []
        acc = []
        loss_uniform = 0

        # compute mask loss
        ratio = self.mask_lambda * ((epoch + 1) * 1.0 / max_epochs)
        if self.cross:
            loss1 = self.get_mask_loss(z1[prototype], mask_out2, ratio)
            loss2 = self.get_mask_loss(z2[prototype], mask_out1, ratio)
        else:
            loss1 = self.get_mask_loss(z1[prototype], mask_out1, ratio)
            loss2 = self.get_mask_loss(z2[prototype], mask_out2, ratio)

        loss.append(loss1)
        loss.append(loss2)

        if self.penalty is not None:
            if self.only_mask_last:
                for i in range(num_mask):
                    loss_uniform += entropy_gradeint(mask_out1[i])
                    loss_uniform += entropy_gradeint(mask_out2[i])
                loss_uniform /= (2 * num_mask)
            else:
                for i in range(n_emb):
                    loss_uniform += entropy_gradeint(z1[i])
                    loss_uniform += entropy_gradeint(z2[i])
                loss_uniform /= (2 * n_emb)

        # compute infonce loss(unimodal and multimodal)
        for i in range(n_emb):
            loss3, acc1 = self.infonce(z1[i], z2[prototype])  # z‘’
            loss4, acc2 = self.infonce(z2[i], z1[prototype])  # z‘
            loss.append((loss3 + loss4) / 2.)
            acc.append((acc1 + acc2) / 2.)
        ssl_acc = {"ssl_acc_%i" % i: acc_ for i, acc_ in enumerate(acc)}
        losses = {"ssl_loss_%i" % i: l for i, l in enumerate(loss)}

        if self.weights is not None:
            loss = torch.mean(torch.stack(loss) * torch.tensor(self.weights, device=z1[0].device))
        else:
            loss = torch.mean(torch.stack(loss))
        if self.penalty is not None:
            loss -= self.penalty * loss_uniform

        acc = torch.mean(torch.stack(acc))

        return {"loss": loss, "ssl_acc": acc, **ssl_acc, **losses}

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)

class InfMasking(nn.Module):
    def __init__(self,
                 in_chans_img: int = 3,
                 in_chans_option: int = 3,
                 in_chans_semantic: int = 3,
                 queue_length: int = 256,
                 batch_size=64,
                 spec=None,
                 ):
        super(InfMasking, self).__init__()  # 移到最前面
        droprate = spec.DROP_RATE
        D_out = spec.OUTPUT_DIM
        num_classes = spec.NUM_CLASSES
        self.freeze_img = spec.FREEZE_IMG

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.img_backbone = nn.Sequential(
            *list(resnet.children())[:-1],  # 去掉最后的 fc 层，输出 B x 2048 x 1 x 1
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()  # 变成 B x 2048
        )
        self.vae = FakeVAE(input_dim=2048, latent_dim=D_out, just_features=True)
        self.img_encoder = nn.Sequential(
            self.img_backbone,
            self.vae,
        )

        self.image_adapter = FeaturesInputAdapter(D_out, D_out)

        self.option_encoder = nn.Sequential(
            nn.Linear(in_chans_option, in_chans_option // 4),
            nn.BatchNorm1d(in_chans_option // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(droprate),

            nn.Linear(in_chans_option // 4, D_out),
            nn.BatchNorm1d(D_out),
            nn.ReLU(inplace=True),
            nn.Dropout(droprate),
        )
        self.ope_adapter = FeaturesInputAdapter(D_out, D_out)

        self.semantic_encoder = nn.Sequential(
            nn.Linear(in_chans_semantic, in_chans_semantic // 4),
            nn.BatchNorm1d(in_chans_semantic // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(droprate),

            nn.Linear(in_chans_semantic // 4, D_out),
            nn.BatchNorm1d(D_out),
            nn.ReLU(inplace=True),
            nn.Dropout(droprate),
        )
        self.emo_adapter = FeaturesInputAdapter(D_out, D_out)

        self.encoders = [
            self.img_encoder,
            self.option_encoder,
            self.semantic_encoder
        ]

        self.adapters = [
            self.image_adapter,
            self.ope_adapter,
            self.emo_adapter
        ]
        self.encoder = InfFusion(
            encoders=self.encoders,
            input_adapters=self.adapters,
            embed_dim = D_out
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(D_out, D_out),
            nn.BatchNorm1d(D_out),
            nn.ReLU(inplace=True),
            nn.Dropout(droprate),
            nn.Linear(D_out, num_classes)
        )

        self.loss = InfMaskingLoss(
            temperature = 0.1,
            mask_lambda = 1
        )

    def freeze_classifier(self):
        print("============Freeze Classifier============")
        for param in self.parameters():
            param.requires_grad = True

        for param in self.classifier.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self):
        print("============Freeze Except Classifier============")
        for param in self.parameters():
            param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = True

    def init_weights(self, zero_init_last: bool = True):
        for n, m in self.named_modules():
            # 如果模块名里含有 'resnet' 就跳过
            if 'img_backbone' in n.lower():
                continue

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_last:
            for n, m in self.named_modules():
                if 'resnet' in n.lower():
                    continue
                if hasattr(m, 'zero_init_last') and callable(m.zero_init_last):
                    m.zero_init_last()

        if self.freeze_img:
            for param in self.img_backbone.parameters():
                param.requires_grad = False

    def forward(
            self,
            v1: Optional[list[torch.Tensor]] = None,
            v2: Optional[list[torch.Tensor]] = None,
            queue_fusion: Optional[torch.Tensor] = None,
            is_train=True,
            is_test=False,
            epoch=1,
            end_epoch=1,
    ):
        all_masks = self.gen_all_possible_masks(len(v1))

        z1, mask_out1 = self.encoder(v1, mask_modalities=all_masks)
        if not is_test:
            z2, mask_out2 = self.encoder(v2, mask_modalities=all_masks)

            outputs = {'aug1_embed': z1,
                    'aug2_embed': z2,
                    'mask_out1': mask_out1,
                    'mask_out2': mask_out2,
                    "prototype": -1,
                    "epoch": epoch,
                    "max_epochs": end_epoch}

            loss_results = self.loss(outputs)
            loss = loss_results['loss']
        else:
            loss = 0.0
        fusion_logits = self.classifier(z1[-1])
        return {
            'fusion_feature': z1[-1],
            'fusion_logits': fusion_logits,
            'intra_cls_loss': loss,
        }
        

    def gen_all_possible_masks(self, n_mod: int):
        """
        :param n_mod: int
        :return: a list of `n_mod` + 1 boolean masks [Mi] such that all but one bool are False.
            A last bool mask is added where all bool are True
        Examples:
        *   For n_mod==2:
            masks == [[True, False], [False, True], [True, True]]
        *   For n_mod == 3:
            masks == [[True, False, False], [False, True, False], [False, False, True], [True, True, True]]
        """
        masks = []
        for L in range(n_mod):
            mask = [s == L for s in range(n_mod)]
            masks.append(mask)
        masks.append([True for _ in range(n_mod)])
        return masks

@register_model
def get_infmasking_model(config, **kwargs):
    model_spec = config.MODEL.SPEC

    infmasking = InfMasking(
        in_chans_img=config.DATASET.IMAGE_CHANS,
        in_chans_option=config.DATASET.OPTION_CHANNEL,
        in_chans_semantic=config.DATASET.SEMANTIC_CHANNEL,
        queue_length=config.QUEUE.QUEUE_LENGTH,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        spec=model_spec
    )

    if config.MODEL.INIT_WEIGHTS:
        infmasking.init_weights(
            zero_init_last=model_spec.ZERO_INIT_LAST
        )

    return infmasking
