from testbed.core.factory import make_network

from .learner import SpectralLearner


def build(*, problem, model_config, method_config, device="cpu", seed=0, method_seed=None, **kwargs):
    network = make_network("resnet_18", problem=problem, model_config=model_config, device=device, seed=seed)
    return SpectralLearner(config=method_config, network=network, problem=problem, seed=seed if method_seed is None else method_seed, **kwargs)
