from torch import nn

from testbed.core.factory import make_network

from .config import CONFIG_CLASS
from .learner import ReDoLearner


def build(*, problem, model_config, method_config, optimizer_config, lr_schedule="constant",
          grad_clip_norm=None, device="cpu", seed=0, start_update=0, method_seed=None):
    config = method_config if isinstance(method_config, CONFIG_CLASS) else CONFIG_CLASS(**method_config)
    network = make_network("vit", problem=problem, model_config=model_config, device=device, seed=seed)
    sites = bind_sites(network, config)
    return ReDoLearner(network=network, config=config, sites=sites, feature_forward=forward_features, problem=problem,
                         optimizer_config=optimizer_config, lr_schedule=lr_schedule, grad_clip_norm=grad_clip_norm,
                         seed=seed if method_seed is None else method_seed, start_update=start_update)


def bind_sites(network, config):
    if config.statistic_site != "after_activation":
        raise ValueError("ViT feedforward recycling uses post-GeLU features; there is no hidden-width normalization")
    if getattr(config, "bias_compensation", False):
        raise ValueError("ViT recycling does not support bias compensation")
    return {f"blocks.{i}.fc1": (block.fc1, nn.Identity(), block.fc2) for i, block in enumerate(network.blocks)}


def forward_features(network, x, statistic_site):
    features = {}
    x = network.embed_tokens(x)
    for i, block in enumerate(network.blocks):
        x = x + block.attention(block.norm1(x))
        activated = block.activation(block.fc1(block.norm2(x)))
        features[f"blocks.{i}.fc1"] = activated.detach().movedim(-1, 1)
        x = x + block.dropout(block.fc2(block.dropout(activated)))
    x = network.norm(x)
    x = x[:, 0] if network.pool == "cls" else x[:, 1:].mean(dim=1)
    for layer in network.head_hidden:
        x = layer(x)
    return network.head(x), features
