from testbed.core.factory import make_network

from .learner import BackpropLearner


def build(*, problem, model_config, method_config, device="cpu", seed=0, method_seed=None, **kwargs):
    network = make_network("mlp", problem=problem, model_config=model_config, device=device, seed=seed)
    return BackpropLearner(network=network, problem=problem, seed=seed if method_seed is None else method_seed, **kwargs)
