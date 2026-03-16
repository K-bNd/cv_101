from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ModelImplem
from timm.layers.mlp import Mlp
from timm.layers.patch_embed import PatchEmbed
from timm.layers.weight_init import trunc_normal_
from timm.models.registry import register_model

from utils import Attention, Block, RMSNorm


class vit_models(
    ModelImplem, pipeline_tag=["image-classification"], tags=["arxiv:2602.08071"]
):
    """Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications from https://github.com/wangf3014/ViT-5/blob/main/models_vit5.py
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        ape=True,
        block_layers=Block,
        Patch_layer=PatchEmbed,
        act_layer=nn.GELU,
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_scale=1e-4,
        flash=True,
        rope=False,
        num_registers=0,
        qk_norm=False,
        reg_theta=10000,
        layer_scale=True,
        **kwargs,
    ):
        super().__init__()
        self.dropout_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_registers = num_registers

        self.patch_embed = Patch_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_token = (
            nn.Parameter(torch.zeros(1, num_registers, embed_dim))
            if num_registers > 0
            else None
        )

        rope_reg_size = int(num_registers**0.5)
        assert rope_reg_size**2 == num_registers, (
            "num_registers must be a square number"
        )

        self.pos_embed = (
            nn.Parameter(torch.zeros(1, num_patches, embed_dim)) if ape else None
        )

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                block_layers(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=0.0,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    Attention_block=Attention_block,
                    Mlp_block=Mlp_block,
                    init_values=init_scale,
                    flash=flash,
                    rope_size=img_size // patch_size if rope else 0,
                    rope_reg_size=rope_reg_size,
                    num_registers=num_registers,
                    qk_norm=qk_norm,
                    reg_theta=reg_theta,
                    layer_scale=layer_scale,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module="head")]
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.cls_token, std=0.02)
        if ape:
            trunc_normal_(self.pos_embed, std=0.02)
        if num_registers > 0:
            trunc_normal_(self.reg_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore  # pyright: ignore[reportArgumentType]
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "reg_token"}

    def get_classifier(self):
        return self.head

    def get_num_layers(self):
        return len(self.blocks)

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        registers = (
            self.reg_token.expand(B, -1, -1) if self.reg_token is not None else None
        )

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = torch.cat((cls_tokens, x), dim=1)
        if registers is not None:
            x = torch.cat((x, registers), dim=1)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):

        x = self.forward_features(x)

        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)

        return x


@register_model
def vit5_small(img_size=224, **kwargs):
    model = vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=False,
        num_registers=4,
        flash=False,
        norm_layer=partial(RMSNorm, eps=1e-6),  # pyright: ignore[reportArgumentType]
        block_layers=Block,
        rope=True,
        rope_reg=True,
        reg_theta=100,
        qk_norm=True,
        **kwargs,
    )
    return model


@register_model
def vit5_base(img_size=224, **kwargs):
    model = vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        num_registers=4,
        flash=False,
        norm_layer=partial(RMSNorm, eps=1e-6),  # pyright: ignore[reportArgumentType]
        block_layers=Block,
        rope=True,
        rope_reg=True,
        reg_theta=100,
        qk_norm=True,
        **kwargs,
    )
    return model


@register_model
def vit5_large(img_size=224, **kwargs):
    model = vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        num_registers=4,
        flash=False,
        norm_layer=partial(RMSNorm, eps=1e-6),  # pyright: ignore[reportArgumentType]
        block_layers=Block,
        rope=True,
        rope_reg=True,
        reg_theta=100,
        qk_norm=True,
        **kwargs,
    )
    return model


@register_model
def vit5_xlarge(img_size=224, **kwargs):
    model = vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        num_registers=4,
        flash=False,
        norm_layer=partial(RMSNorm, eps=1e-6),  # pyright: ignore[reportArgumentType]
        block_layers=Block,
        rope=True,
        rope_reg=True,
        reg_theta=100,
        qk_norm=True,
        **kwargs,
    )
    return model
