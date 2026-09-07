from testbed.core.factory import make_network

from .config import CONFIG_CLASS
from .learner import ReDoLearner


def build(*, problem, model_config, method_config, optimizer_config, lr_schedule="constant",
          grad_clip_norm=None, device="cpu", seed=0, start_update=0, method_seed=None):
    config = method_config if isinstance(method_config, CONFIG_CLASS) else CONFIG_CLASS(**method_config)
    network = make_network("mlp", problem=problem, model_config=model_config, device=device, seed=seed)
    sites = bind_sites(network, config)
    return ReDoLearner(network=network, config=config, sites=sites, feature_forward=forward_features, problem=problem,
                         optimizer_config=optimizer_config, lr_schedule=lr_schedule, grad_clip_norm=grad_clip_norm,
                         seed=seed if method_seed is None else method_seed, start_update=start_update)


def bind_sites(network, config):
    layers = [(f"hidden.{i}", layer) for i, layer in enumerate(network.hidden)]
    layers += [(f"head_hidden.{i}", layer) for i, layer in enumerate(network.head_hidden)]
    if getattr(config, "bias_compensation", False):
        if any(layer.norm_position != "pre_activation" or layer.dropout.p for _, layer in layers):
            raise ValueError("MLP bias compensation requires pre-activation normalization and no dropout")
    sites = {name: (layer.linear, layer.norm, layers[i + 1][1].linear if i + 1 < len(layers) else network.head)
             for i, (name, layer) in enumerate(layers)}
    if not sites:
        raise ValueError("unit recycling requires at least one hidden layer")
    return sites


def forward_features(network, x, statistic_site):
    features = {}
    x = x.flatten(1)
    layers = [(f"hidden.{i}", layer) for i, layer in enumerate(network.hidden)]
    layers += [(f"head_hidden.{i}", layer) for i, layer in enumerate(network.head_hidden)]
    for name, layer in layers:
        x = layer.linear(x)
        if layer.norm_position == "pre_activation":
            normalized = layer.norm(x)
            activated = layer.activation(normalized)
            x = activated
        else:
            activated = layer.activation(x)
            normalized = layer.norm(activated)
            x = normalized
        features[name] = (activated if statistic_site == "after_activation" else normalized).detach()
        x = layer.dropout(x)
    return network.head(x), features
