# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import collections.abc
import math
import os
import re
import sys
import warnings
from collections import OrderedDict
from functools import partial
from itertools import repeat

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from .layers_quant import DropPath, HybridEmbed, Mlp, PatchEmbed, trunc_normal_
from .ptq import QAct, QConv2d, QIntLayerNorm, QIntSoftmax, QLinear, QIntSoftmaxUniform, QIntGELU, QIntSoftmaxShift, QIntGELUShift, QIntSoftermax, QIntSoftermaxLinear
from .utils import load_weights_from_npz

import wandb

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'vit_base_patch16_224', 'vit_large_patch16_224'
]


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 quant=False,
                 calibrate=False,
                 cfg=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = QLinear(dim,
                           dim * 3,
                           bias=qkv_bias,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=cfg.BIT_TYPE_W,
                           calibration_mode=cfg.CALIBRATION_MODE_W,
                           observer_str=cfg.OBSERVER_W,
                           quantizer_str=cfg.QUANTIZER_W)
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.proj = QLinear(dim,
                            dim,
                            quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W)
        self.qact3 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.qact_attn1 = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.log_int_softmax = getattr(sys.modules[__name__], cfg.softmax_type)(
            log_int_softmax=cfg.INT_SOFTMAX,
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_S,
            calibration_mode=cfg.CALIBRATION_MODE_S,
            observer_str=cfg.OBSERVER_S,
            quantizer_str=cfg.QUANTIZER_S
        )
        print(f'softmax_type: {cfg.softmax_type}')

    def forward(self, x, mask = None, mask_softmax_bias = -1000.):
        B, N, C = x.shape
        # import ipdb; ipdb.set_trace()
        x = self.qkv(x)
        x = self.qact1(x)
        qkv = x.reshape(B, N, 3, self.num_heads,
                        C // self.num_heads).permute(2, 0, 3, 1, 4)  # (BN33)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.qact_attn1(attn)

        # ---- for act only ----
        if mask is not None:
            attn = attn + mask.view(mask.shape[0], 1, 1, mask.shape[1]) * mask_softmax_bias
        # ---- for act only end ----
        attn = self.log_int_softmax(attn, self.qact_attn1.quantizer.scale)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.qact2(x)
        x = self.proj(x)
        x = self.qact3(x)
        x = self.proj_drop(x)
        return x

        '''
        关于此处的mask需要注意: mask给attn带来了极大的异常值，导致后一步attn量化的计算不正常，所以应该对mask和quant的位置稍作调整
        '''


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 quant=False,
                 calibrate=False,
                 cfg=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              cfg=cfg)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                          observer_str=cfg.OBSERVER_A_LN,
                          quantizer_str=cfg.QUANTIZER_A_LN)
        self.norm2 = norm_layer(dim)
        self.qact3 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop,
                       quant=quant,
                       calibrate=calibrate,
                       cfg=cfg)
        self.qact4 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                          observer_str=cfg.OBSERVER_A_LN,
                          quantizer_str=cfg.QUANTIZER_A_LN)

        self.gate_scale = cfg.gate_scale
        self.gate_center = cfg.gate_center

        print(f'gate_scale: {self.gate_scale}, gate_center: {self.gate_center}')

    def forward(self, x, last_quantizer=None, mask=None):
        bs, token, dim = x.shape

        x = self.qact2(x + self.drop_path(
            self.attn(
                self.qact1(self.norm1(x*(1-mask).view(bs, token, 1), last_quantizer,
                                      self.qact1.quantizer)*(1-mask).view(bs, token, 1)), mask = mask)))
        x = self.qact4(x + self.drop_path(
            self.mlp(
                self.qact3(
                    self.norm2(x*(1-mask).view(bs, token, 1), self.qact2.quantizer,
                               self.qact3.quantizer)*(1-mask).view(bs, token, 1)))))

        halting_score_token = torch.sigmoid(self.gate_scale * x[:,:,0] - self.gate_center)
        halting_score = [-1, halting_score_token]

        return x, halting_score


class VisionTransformer(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 representation_size=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 hybrid_backbone=None,
                 norm_layer=None,
                 quant=False,
                 calibrate=False,
                 input_quant=False,
                 cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.cfg = cfg
        self.input_quant = input_quant
        if input_quant:
            self.qact_input = QAct(quant=quant,
                                   calibrate=calibrate,
                                   bit_type=cfg.BIT_TYPE_A,
                                   calibration_mode=cfg.CALIBRATION_MODE_A,
                                   observer_str=cfg.OBSERVER_A,
                                   quantizer_str=cfg.QUANTIZER_A)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed(img_size=img_size,
                                          patch_size=patch_size,
                                          in_chans=in_chans,
                                          embed_dim=embed_dim,
                                          quant=quant,
                                          calibrate=calibrate,
                                          cfg=cfg)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.qact_embed = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)
        self.qact_pos = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                          observer_str=cfg.OBSERVER_A_LN,
                          quantizer_str=cfg.QUANTIZER_A_LN)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer,
                  quant=quant,
                  calibrate=calibrate,
                  cfg=cfg) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict([
                    ('fc', nn.Linear(embed_dim, representation_size)),
                    ('act', nn.Tanh()),
                ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = (QLinear(self.num_features,
                             num_classes,
                             quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_W,
                             calibration_mode=cfg.CALIBRATION_MODE_W,
                             observer_str=cfg.OBSERVER_W,
                             quantizer_str=cfg.QUANTIZER_W)
                     if num_classes > 0 else nn.Identity())
        self.act_out = QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_A,
                            calibration_mode=cfg.CALIBRATION_MODE_A,
                            observer_str=cfg.OBSERVER_A,
                            quantizer_str=cfg.QUANTIZER_A)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        # special for quantization
        self.gate_scale = 10
        self.gate_center = 75
        num_patches = self.patch_embed.num_patches

        print('\nNow this is an ACT DeiT.\n')
        self.eps = 0.01
        print(f'Setting eps as {self.eps}.')

        print('Now setting up the rho.')
        self.rho = None  # Ponder cost
        self.counter = None  # Keeps track of how many layers are used for each example (for logging)
        self.batch_cnt = 0 # amount of batches seen, mainly for tensorboard

        # for token act part
        self.c_token = None
        self.R_token = None
        self.mask_token = None
        self.rho_token = None
        self.counter_token = None
        self.num_tokens = 1
        self.total_token_cnt = num_patches + self.num_tokens

        self.step = 0

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = (nn.Linear(self.embed_dim, num_classes)
                     if num_classes > 0 else nn.Identity())

    def model_quant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QIntSoftmaxUniform, QIntGELU, QIntSoftmaxShift, QIntGELUShift]:
                m.quant = True
            if self.cfg.INT_NORM:
                if type(m) in [QIntLayerNorm]:
                    m.mode = 'int'

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QIntSoftmaxUniform, QIntGELU, QIntSoftmaxShift, QIntGELUShift]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QIntSoftmaxUniform, QIntGELU, QIntSoftmaxShift, QIntGELUShift]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QIntSoftmaxUniform, QIntGELU, QIntSoftmaxShift, QIntGELUShift]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QIntSoftmaxUniform, QIntGELU, QIntSoftmaxShift, QIntGELUShift]:
                m.calibrate = False

    # def forward_features(self, x):
    #     B = x.shape[0]

    #     if self.input_quant:
    #         x = self.qact_input(x)

    #     x = self.patch_embed(x)

    #     cls_tokens = self.cls_token.expand(
    #         B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     x = self.qact_embed(x)
    #     x = x + self.qact_pos(self.pos_embed)
    #     x = self.qact1(x)

    #     x = self.pos_drop(x)

    #     for i, blk in enumerate(self.blocks):
    #         last_quantizer = self.qact1.quantizer if i == 0 else self.blocks[i - 1].qact4.quantizer
    #         x = blk(x, last_quantizer)

    #     x = self.norm(x, self.blocks[-1].qact4.quantizer,
    #                   self.qact2.quantizer)[:, 0]
    #     x = self.qact2(x)
    #     x = self.pre_logits(x)
    #     return x

    def forward_features(self, x):
        B = x.shape[0]
        bs = B

        if self.input_quant:
            x = self.qact_input(x)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.qact_embed(x)
        x = x + self.qact_pos(self.pos_embed)
        x = self.qact1(x)

        x = self.pos_drop(x)

        # ---- for act only ----
        if self.c_token is None or B != self.c_token.size()[0]:
            self.c_token = Variable(torch.zeros(B, self.total_token_cnt).cuda())
            self.R_token = Variable(torch.ones(B, self.total_token_cnt).cuda())
            self.mask_token = Variable(torch.ones(B, self.total_token_cnt).cuda())
            self.rho_token = Variable(torch.zeros(B, self.total_token_cnt).cuda())
            self.counter_token = Variable(torch.ones(B, self.total_token_cnt).cuda())
        
        c_token = self.c_token.clone() #! [10, 197] {line 2}
        R_token = self.R_token.clone() #! [10, 197] Remainder value {line 3}
        mask_token = self.mask_token.clone() #! [10, 197] Token mask {line 6}
        self.rho_token = self.rho_token.detach() * 0. #! Token ponder loss vector {line 5}
        self.counter_token = self.counter_token.detach() * 0 + 1. #! [10, 197]
        # Will contain the output of this residual layer (weighted sum of outputs of the residual blocks)
        output = None
        # Use out to backbone
        out = x
        self.halting_score_layer = []
        # ---- for act only end----

        for i, blk in enumerate(self.blocks):
            last_quantizer = self.qact1.quantizer if i == 0 else self.blocks[i - 1].qact4.quantizer
            # ---- for act only ----
            out.data = out.data * mask_token.float().view(B, self.total_token_cnt, 1) #! [10, 197, 192] * [10, 197, 1] {line 8}
            
            # ---- for act only end ----
            block_output, h_lst = blk(out, last_quantizer, 1.- mask_token.float())

            # ---- for act only ----
            self.halting_score_layer.append(torch.mean(h_lst[1][1:]))
            out = block_output.clone()
            _, h_token = h_lst
            block_output = block_output * mask_token.float().view(B, self.total_token_cnt, 1) #! [10, 197] -> [10, 197, 1]

            if i == len(self.blocks) - 1:
                h_token = Variable(torch.ones(bs, self.total_token_cnt).cuda()) 

            c_token = c_token + h_token #! {line 14}
            self.rho_token = self.rho_token + mask_token.float() #! {line 15}

            # case 1
            reached_token = c_token > 1 - self.eps #! {line 17} #! [10, 197]
            if self.step % 10 == 0:
                print(f'avg_val_{i}', torch.mean(c_token).item())
                wandb.log({f'avg_val_{i}': torch.mean(c_token).item()})
                print(f"reached_token_ratio_{i}", torch.mean(reached_token.float()).item())
                wandb.log({f"reached_token_ratio_{i}": torch.mean(reached_token.float()).item()})
            reached_token = reached_token.float() * mask_token.float() #! 抽取出本轮达到目标值的token，同时还要忽略掉之前被mask掉的token
            delta1 = block_output * R_token.view(bs, self.total_token_cnt, 1) * reached_token.view(bs, self.total_token_cnt, 1) #! {line 26}
            self.rho_token = self.rho_token + R_token * reached_token #! {line 20}

            # case 2
            not_reached_token = c_token < 1 - self.eps
            not_reached_token = not_reached_token.float() #! the masked token is directy included in the range
            R_token = R_token - (not_reached_token.float() * h_token) #! {line 18}
            delta2 = block_output * h_token.view(bs, self.total_token_cnt, 1) * not_reached_token.view(bs, self.total_token_cnt, 1) #! {line 24}

            self.counter_token = self.counter_token + not_reached_token # These data points will need at least one more layer

            # Update the mask
            mask_token = c_token < 1 - self.eps #! {line 28}

            if output is None:
                output = delta1 + delta2
            else:
                output = output + (delta1 + delta2)

            # ---- for act only end ----

        x = self.norm(output, self.blocks[-1].qact4.quantizer, self.qact2.quantizer)[:, 0]
        x = self.qact2(x)
        x = self.pre_logits(x)
        return x

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     x = self.head(x)
    #     x = self.act_out(x)
    #     return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = self.act_out(x)
        self.step += 1
        return x


def deit_tiny_patch16_224(pretrained=False,
                          quant=False,
                          calibrate=False,
                          cfg=None,
                          **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(QIntLayerNorm, eps=1e-6),
        quant=quant,
        calibrate=calibrate,
        input_quant=True,
        cfg=cfg,
        **kwargs,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=
            'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
            map_location='cpu',
            check_hash=True,
        )
        model.load_state_dict(checkpoint['model'], strict=False)
    return model


def deit_small_patch16_224(pretrained=False,
                           quant=False,
                           calibrate=False,
                           cfg=None,
                           **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=384,
                              depth=12,
                              num_heads=6,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(QIntLayerNorm, eps=1e-6),
                              quant=quant,
                              calibrate=calibrate,
                              input_quant=True,
                              cfg=cfg,
                              **kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=
            'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
            map_location='cpu',
            check_hash=True)
        model.load_state_dict(checkpoint['model'], strict=False)
    return model


def deit_base_patch16_224(pretrained=False,
                          quant=False,
                          calibrate=False,
                          cfg=None,
                          **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(QIntLayerNorm, eps=1e-6),
                              quant=quant,
                              calibrate=calibrate,
                              input_quant=True,
                              cfg=cfg,
                              **kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=
            'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
            map_location='cpu',
            check_hash=True)
        model.load_state_dict(checkpoint['model'], strict=False)
    return model


def vit_base_patch16_224(pretrained=False,
                         quant=False,
                         calibrate=False,
                         cfg=None,
                         **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(QIntLayerNorm, eps=1e-6),
                              quant=quant,
                              calibrate=calibrate,
                              input_quant=False,
                              cfg=cfg,
                              **kwargs)
    if pretrained:
        url = 'https://storage.googleapis.com/vit_models/augreg/' + \
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'

        load_weights_from_npz(model, url, check_hash=True)
    return model


def vit_large_patch16_224(pretrained=False,
                          quant=False,
                          calibrate=False,
                          cfg=None,
                          **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(QIntLayerNorm, eps=1e-6),
                              quant=quant,
                              calibrate=calibrate,
                              input_quant=False,
                              cfg=cfg,
                              **kwargs)
    if pretrained:
        url = 'https://storage.googleapis.com/vit_models/augreg/' + \
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'

        load_weights_from_npz(model, url, check_hash=True)
    return model
