import torch
import torch.nn as nn
from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp

try:
    from flash_attn import flash_attn_qkvpacked_func
except ImportError:
    pass

from rope import VisionRotaryEmbedding


class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        flash=True,
        rope_size=0,
        rope_reg_size=0,
        num_registers=0,
        reg_theta=10000,
        qk_norm=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.flash = flash
        self.num_registers = num_registers
        self.rope = (
            VisionRotaryEmbedding(head_dim // 2, rope_size) if rope_size > 0 else None
        )
        self.rope_reg = (
            VisionRotaryEmbedding(head_dim // 2, rope_reg_size, theta=reg_theta)
            if rope_reg_size > 0
            else None
        )

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def forward(self, x):
        B, N, C = x.shape
        reg_idx = N - self.num_registers

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)

        if self.qk_norm:
            qk_dtype = q.dtype
            q = self.q_norm(q).to(qk_dtype)
            k = self.k_norm(k).to(qk_dtype)

        if self.rope is not None:
            q = torch.cat((q[:, :1], self.rope(q[:, 1:reg_idx]), q[:, reg_idx:]), dim=1)
            k = torch.cat((k[:, :1], self.rope(k[:, 1:reg_idx]), k[:, reg_idx:]), dim=1)
        if self.rope_reg is not None:
            q = torch.cat(
                (q[:, :1], q[:, 1:reg_idx], self.rope_reg(q[:, reg_idx:])), dim=1
            )
            k = torch.cat(
                (k[:, :1], k[:, 1:reg_idx], self.rope_reg(k[:, reg_idx:])), dim=1
            )

        if self.flash:
            qkv = torch.stack([q, k, v], dim=2)
            x = flash_attn_qkvpacked_func(qkv).reshape(B, N, C)  # pyright: ignore[reportPossiblyUnboundVariable]
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        drop=0.0,
        norm_layer=nn.LayerNorm,
        subln=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
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
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_values=1e-4,
        flash=True,
        rope_size=0,
        rope_reg_size=0,
        reg_theta=10000,
        num_registers=0,
        qk_norm=False,
        layer_scale=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            flash=flash,
            rope_size=rope_size,
            rope_reg_size=rope_reg_size,
            num_registers=num_registers,
            qk_norm=qk_norm,
            reg_theta=reg_theta,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )

    def forward(self, x):
        if self.layer_scale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
