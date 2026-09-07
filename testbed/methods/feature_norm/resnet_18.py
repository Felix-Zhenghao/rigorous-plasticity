from testbed.core.factory import make_network

from .learner import FeatureNormLearner


def forward_features(network, x, sites):
    features = {}
    x = network.stem_pool(network.stem_activation(network.stem_norm(network.stem_conv(x))))
    if "stem" in sites:
        features["stem"] = x
    for stage_index, stage in enumerate(network.stages):
        for block_index, block in enumerate(stage):
            residual = block.shortcut(x)
            x = block.activation1(block.norm1(block.conv1(x)))
            name = f"stages.{stage_index}.{block_index}.activation1"
            if name in sites:
                features[name] = x
            x = block.activation2(block.sum_norm(block.norm2(block.conv2(x)) + residual))
            name = f"stages.{stage_index}.{block_index}.activation2"
            if name in sites:
                features[name] = x
    x = x.mean(dim=(2, 3))
    if "pooled" in sites:
        features["pooled"] = x
    for index, layer in enumerate(network.head_hidden):
        x = layer(x)
        name = f"head_hidden.{index}"
        if name in sites:
            features[name] = x
    if "head_input" in sites:
        features["head_input"] = x
    return network.head(x), features


def build(*, problem, model_config, method_config, device="cpu", seed=0, method_seed=None, **kwargs):
    network = make_network("resnet_18", problem=problem, model_config=model_config, device=device, seed=seed)
    available = {"head_input", "pooled", "stem"} | {f"head_hidden.{i}" for i in range(len(network.head_hidden))} | {f"stages.{s}.{b}.activation{a}" for s, stage in enumerate(network.stages) for b in range(len(stage)) for a in (1, 2)}
    if not set(method_config.feature_sites) <= available:
        raise ValueError(f"unknown feature sites; available: {sorted(available)}")
    return FeatureNormLearner(config=method_config, forward_features=forward_features,
                              network=network, problem=problem, seed=seed if method_seed is None else method_seed, **kwargs)
