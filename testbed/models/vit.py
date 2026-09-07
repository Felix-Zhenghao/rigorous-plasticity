from types import SimpleNamespace

import torch
from torch import nn

from .layers import HiddenLayer, layer_norm


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.q = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.k = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.v = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.out = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, x):
        batch, tokens, _ = x.shape
        q, k, v = [layer(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
                   for layer in (self.q, self.k, self.v)]
        scores = (q @ k.transpose(-1, -2)) * self.head_dim**-0.5
        mixed = self.dropout(scores.softmax(dim=-1)) @ v
        return self.out(mixed.transpose(1, 2).reshape(batch, tokens, -1))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = layer_norm(config.embed_dim, config)
        self.attention = Attention(config)
        self.norm2 = layer_norm(config.embed_dim, config)
        self.fc1 = nn.Linear(config.embed_dim, config.mlp_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(config.mlp_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.feedforward_dropout)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        hidden = self.dropout(self.activation(self.fc1(self.norm2(x))))
        return x + self.dropout(self.fc2(hidden))


class ViT(nn.Module):
    def __init__(self, problem, config):
        super().__init__()
        if len(problem.input_shape) != 3:
            raise ValueError("ViT requires CHW inputs")
        channels, height, width = problem.input_shape
        patch = config.patch_size
        if height % patch or width % patch:
            raise ValueError("input dimensions must be divisible by patch_size")
        self.patch_size = patch
        self.patch_embedding = config.patch_embedding
        patch_dim = channels * patch * patch
        self.patch_norm = layer_norm(patch_dim, config) if config.normalization_layout == "patch_norm" else nn.Identity()
        if config.patch_embedding == "conv":
            if config.normalization_layout == "patch_norm":
                raise ValueError("patch_norm layout requires linear patch embedding")
            self.patch_embed = nn.Conv2d(channels, config.embed_dim, patch, stride=patch)
        else:
            self.patch_embed = nn.Linear(patch_dim, config.embed_dim)
        self.embedding_norm = layer_norm(config.embed_dim, config) if config.normalization_layout == "patch_norm" else nn.Identity()
        token_count = height // patch * (width // patch)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.embed_dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, token_count + 1, config.embed_dim))
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.depth)])
        self.norm = layer_norm(config.embed_dim, config)
        self.pool = config.pool
        self.head_hidden = nn.ModuleList()
        head_config = SimpleNamespace(bias=True, norm="none", activation="gelu", negative_slope=0.01,
                                      norm_position="pre_activation", dropout=config.feedforward_dropout)
        width = config.embed_dim
        for size in config.head_hidden_sizes:
            self.head_hidden.append(HiddenLayer(width, size, head_config))
            width = size
        self.head_input_dim = width
        self.head = nn.Linear(width, len(problem.output_ids), bias=config.head_bias)

    def embed_tokens(self, x):
        if self.patch_embedding == "conv":
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
        else:
            patch = self.patch_size
            batch, channels, height, width = x.shape
            x = x.reshape(batch, channels, height // patch, patch, width // patch, patch)
            x = x.permute(0, 2, 4, 1, 3, 5).reshape(batch, -1, channels * patch * patch)
            x = self.patch_embed(self.patch_norm(x))
        x = self.embedding_norm(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        return self.embedding_dropout(x + self.pos_embedding)

    def forward(self, x):
        x = self.embed_tokens(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x[:, 0] if self.pool == "cls" else x[:, 1:].mean(dim=1)
        for layer in self.head_hidden:
            x = layer(x)
        return self.head(x)


def build(problem, config):
    return ViT(problem, config)
