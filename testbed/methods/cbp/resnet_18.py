from testbed.core.factory import make_network

from .config import CONFIG_CLASS
from .learner import CBPLearner


def build(*, problem, model_config, method_config, optimizer_config, lr_schedule="constant",
          grad_clip_norm=None, device="cpu", seed=0, start_update=0, method_seed=None):
    config = method_config if isinstance(method_config, CONFIG_CLASS) else CONFIG_CLASS(**method_config)
    network = make_network("resnet_18", problem=problem, model_config=model_config, device=device, seed=seed)
    sites = bind_sites(network, config)
    return CBPLearner(network=network, config=config, sites=sites, feature_forward=forward_features, problem=problem,
                         optimizer_config=optimizer_config, lr_schedule=lr_schedule, grad_clip_norm=grad_clip_norm,
                         seed=seed if method_seed is None else method_seed, start_update=start_update)


def bind_sites(network, config):
    if getattr(config, "bias_compensation", False):
        raise ValueError("ResNet recycling does not support bias compensation")
    return {f"stages.{i}.{j}.conv1": (block.conv1, block.norm1, block.conv2)
            for i, stage in enumerate(network.stages) for j, block in enumerate(stage)}


def forward_features(network, x, statistic_site):
    features = {}
    x = network.stem_pool(network.stem_activation(network.stem_norm(network.stem_conv(x))))
    for i, stage in enumerate(network.stages):
        for j, block in enumerate(stage):
            residual = block.shortcut(x)
            normalized = block.norm1(block.conv1(x))
            activated = block.activation1(normalized)
            features[f"stages.{i}.{j}.conv1"] = (activated if statistic_site == "after_activation" else normalized).detach()
            x = block.activation2(block.sum_norm(block.norm2(block.conv2(activated)) + residual))
    x = x.mean(dim=(2, 3))
    for layer in network.head_hidden:
        x = layer(x)
    return network.head(x), features
