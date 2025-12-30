import torch.nn.functional as func
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Tuple
from torchvision import models

from . import FakeVAE
from .register import register_model

def mlp_head(dim_in, feat_dim):
    return nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
def mlp(dim, hidden_dim, output_dim, layers, activation):
    activation = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)

class InfoNCECritic(nn.Module):
    def __init__(self, A_dim, B_dim, hidden_dim, layers, activation, **extra_kwargs):
        super(InfoNCECritic, self).__init__()
        # output is scalar score
        self._f = mlp(A_dim + B_dim, hidden_dim, 1, layers, activation)

    def forward(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self._f(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self._f(torch.cat([x_tile, y_tile], dim=-1))

        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - np.log(sample_size))
        return -lower_bound


# Concat critic with the CLUBInfoNCE (NCE-CLUB) objective
class CLUBInfoNCECritic(nn.Module):
    def __init__(self, A_dim, B_dim, hidden_dim, layers, activation, **extra_kwargs):
        super(CLUBInfoNCECritic, self).__init__()

        self._f = mlp(A_dim + B_dim, hidden_dim, 1, layers, activation)

    # CLUB loss
    def forward(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self._f(torch.cat([y_samples, x_samples], dim=-1))
        T1 = self._f(torch.cat([y_tile, x_tile], dim=-1))

        return T0.mean() - T1.mean()

    # InfoNCE loss
    def learning_loss(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self._f(torch.cat([y_samples, x_samples], dim=-1))
        T1 = self._f(torch.cat([y_tile, x_tile], dim=-1))

        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - np.log(sample_size))
        return -lower_bound

class FactorCLSSL(nn.Module):
    def __init__(self, encoders, feat_dims, y_ohe_dim, temperature=1, activation='relu', lr=1e-4, ratio=1):
        super(FactorCLSSL, self).__init__()
        self.critic_hidden_dim = 512
        self.critic_layers = 1
        self.critic_activation = 'relu'
        self.lr = lr
        self.ratio = ratio
        self.y_ohe_dim = y_ohe_dim
        self.temperature = temperature

        # encoder backbones
        self.feat_dims = feat_dims
        self.backbones = nn.ModuleList(encoders)

        # linear projection heads
        self.linears_infonce_x1x2 = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])
        self.linears_club_x1x2_cond = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])

        self.linears_infonce_x1y = mlp_head(self.feat_dims[0], self.feat_dims[0])
        self.linears_infonce_x2y = mlp_head(self.feat_dims[1], self.feat_dims[1])
        self.linears_infonce_x1x2_cond = nn.ModuleList(
            [mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])
        self.linears_club_x1x2 = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])

        # critics
        self.infonce_x1x2 = InfoNCECritic(self.feat_dims[0], self.feat_dims[1], self.critic_hidden_dim,
                                          self.critic_layers, activation, temperature=temperature)
        self.club_x1x2_cond = CLUBInfoNCECritic(self.feat_dims[0] * 2, self.feat_dims[1] * 2,
                                                self.critic_hidden_dim, self.critic_layers, activation,
                                                temperature=temperature)

        self.infonce_x1y = InfoNCECritic(self.feat_dims[0], self.feat_dims[0], self.critic_hidden_dim,
                                         self.critic_layers, activation, temperature=temperature)
        self.infonce_x2y = InfoNCECritic(self.feat_dims[1], self.feat_dims[1], self.critic_hidden_dim,
                                         self.critic_layers, activation, temperature=temperature)
        self.infonce_x1x2_cond = InfoNCECritic(self.feat_dims[0] * 2, self.feat_dims[1] * 2,
                                               self.critic_hidden_dim, self.critic_layers, activation,
                                               temperature=temperature)
        self.club_x1x2 = CLUBInfoNCECritic(self.feat_dims[0], self.feat_dims[1], self.critic_hidden_dim,
                                           self.critic_layers, activation, temperature=temperature)

    def ohe(self, y):
        N = y.shape[0]
        y_ohe = torch.zeros((N, self.y_ohe_dim))
        y_ohe[torch.arange(N).long(), y.T[0].long()] = 1
        return y_ohe

    def forward(self, x1, x2, x1_aug, x2_aug):
        # Get embeddings
        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)

        x1_aug_embed = self.backbones[0](x1_aug)
        x2_aug_embed = self.backbones[1](x2_aug)

        # compute losses
        uncond_losses = [
            self.infonce_x1x2(self.linears_infonce_x1x2[0](x1_embed), self.linears_infonce_x1x2[1](x2_embed)),
            self.club_x1x2(self.linears_club_x1x2[0](x1_embed), self.linears_club_x1x2[1](x2_embed)),
            self.infonce_x1y(self.linears_infonce_x1y(x1_embed), self.linears_infonce_x1y(x1_aug_embed)),
            self.infonce_x2y(self.linears_infonce_x2y(x2_embed), self.linears_infonce_x2y(x2_aug_embed))
            ]

        cond_losses = [self.infonce_x1x2_cond(torch.cat([self.linears_infonce_x1x2_cond[0](x1_embed),
                                                         self.linears_infonce_x1x2_cond[0](x1_aug_embed)], dim=1),
                                              torch.cat([self.linears_infonce_x1x2_cond[1](x2_embed),
                                                         self.linears_infonce_x1x2_cond[1](x2_aug_embed)], dim=1)),
                       self.club_x1x2_cond(torch.cat([self.linears_club_x1x2_cond[0](x1_embed),
                                                      self.linears_club_x1x2_cond[0](x1_aug_embed)], dim=1),
                                           torch.cat([self.linears_club_x1x2_cond[1](x2_embed),
                                                      self.linears_club_x1x2_cond[1](x2_aug_embed)], dim=1))
                       ]

        return sum(uncond_losses) + sum(cond_losses)

    def learning_loss(self, x1, x2, x1_aug, x2_aug):
        # Get embeddings
        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)

        x1_aug_embed = self.backbones[0](x1_aug)
        x2_aug_embed = self.backbones[1](x2_aug)

        # Calculate InfoNCE loss for CLUB-NCE
        learning_losses = [
            self.club_x1x2.learning_loss(self.linears_club_x1x2[0](x1_embed), self.linears_club_x1x2[1](x2_embed)),
            self.club_x1x2_cond.learning_loss(torch.cat([self.linears_club_x1x2_cond[0](x1_embed),
                                                         self.linears_club_x1x2_cond[0](x1_aug_embed)], dim=1),
                                              torch.cat([self.linears_club_x1x2_cond[1](x2_embed),
                                                         self.linears_club_x1x2_cond[1](x2_aug_embed)], dim=1))
            ]
        return sum(learning_losses)

    def get_embedding(self, x1, x2, just_x1=False):
        x1_embed = self.backbones[0](x1)
        x1_reps = [self.linears_infonce_x1x2[0](x1_embed),
                   self.linears_club_x1x2[0](x1_embed),
                   self.linears_infonce_x1y(x1_embed),
                   self.linears_infonce_x1x2_cond[0](x1_embed),
                   self.linears_club_x1x2_cond[0](x1_embed)]

        x2_embed = self.backbones[1](x2)
        x2_reps = [self.linears_infonce_x1x2[1](x2_embed),
                   self.linears_club_x1x2[1](x2_embed),
                   self.linears_infonce_x2y(x2_embed),
                   self.linears_infonce_x1x2_cond[1](x2_embed),
                   self.linears_club_x1x2_cond[1](x2_embed)]

        return torch.cat(x1_reps, dim=1), torch.cat(x2_reps, dim=1)


class FactorCL(nn.Module):
    def __init__(self,
                 in_chans_img: int = 3,
                 in_chans_option: int = 3,
                 in_chans_semantic: int = 3,
                 queue_length: int = 256,
                 batch_size=64,
                 spec=None,
                 ):
        super(FactorCL, self).__init__()  # 移到最前面
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

        self.encoders = [
            self.img_encoder,
            self.semantic_encoder
        ]

        self.feat_dims = [D_out, D_out]

        self.classifier = nn.Sequential(
            nn.Linear(D_out*(5 * 2), D_out),
            nn.BatchNorm1d(D_out),
            nn.ReLU(inplace=True),
            nn.Dropout(droprate),
            nn.Linear(D_out, num_classes)
        )

        self.factorcl_ssl = FactorCLSSL(
            encoders=self.encoders,
            feat_dims=self.feat_dims,
            y_ohe_dim=3,
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
        x1, x2 = v1[0], v1[1]
        if not is_test:
            x1_aug, x2_aug = v2[0], v2[1]

            loss = self.factorcl_ssl(x1, x2, x1_aug, x2_aug)
        else:
            loss = 0.0
        z1, z2 = self.factorcl_ssl.get_embedding(x1, x2)
        z = torch.cat([z1, z2], dim=1)
        fusion_logits = self.classifier(z)
        return {
            'fusion_feature': z1[-1],
            'fusion_logits': fusion_logits,
            'intra_cls_loss': loss,
        }


@register_model
def get_factorcl_model(config, **kwargs):
    model_spec = config.MODEL.SPEC

    factorcl = FactorCL(
        in_chans_img=config.DATASET.IMAGE_CHANS,
        in_chans_option=config.DATASET.OPTION_CHANNEL,
        in_chans_semantic=config.DATASET.SEMANTIC_CHANNEL,
        queue_length=config.QUEUE.QUEUE_LENGTH,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        spec=model_spec
    )

    if config.MODEL.INIT_WEIGHTS:
        factorcl.init_weights(
            zero_init_last=model_spec.ZERO_INIT_LAST
        )

    return factorcl
