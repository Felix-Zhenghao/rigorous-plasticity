from dataclasses import asdict, is_dataclass

from testbed.core.factory import make_network
from testbed.methods.backprop.learner import BackpropLearner


def build(*, problem, model_config, method_config, device="cpu", seed=0, method_seed=None, **kwargs):
    values = asdict(model_config) if is_dataclass(model_config) else dict(model_config)
    values["activation"] = "leaky_relu"
    if "negative_slope" in values and method_config.negative_slope != 0.01 and values["negative_slope"] != method_config.negative_slope:
        raise ValueError("model and method negative_slope disagree")
    values.setdefault("negative_slope", method_config.negative_slope)
    network = make_network("resnet_18", problem=problem, model_config=values, device=device, seed=seed)
    return BackpropLearner(network=network, problem=problem, seed=seed if method_seed is None else method_seed, **kwargs)
