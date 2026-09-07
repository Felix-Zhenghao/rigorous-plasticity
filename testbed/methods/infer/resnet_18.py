from testbed.core.factory import make_network

from .learner import InFeRLearner


def forward_features(network, x):
    x = network.stem_pool(network.stem_activation(network.stem_norm(network.stem_conv(x))))
    for stage in network.stages:
        x = stage(x)
    x = x.mean(dim=(2, 3))
    for layer in network.head_hidden:
        x = layer(x)
    return network.head(x), x


def build(*, problem, model_config, method_config, device="cpu", seed=0, method_seed=None, **kwargs):
    network = make_network("resnet_18", problem=problem, model_config=model_config, device=device, seed=seed)
    return InFeRLearner(config=method_config, forward_features=forward_features,
                        network=network, problem=problem, seed=seed if method_seed is None else method_seed, **kwargs)
