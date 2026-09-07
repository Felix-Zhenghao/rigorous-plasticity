from testbed.core.factory import make_network

from .config import CONFIG_CLASS
from .learner import CChainLearner


def build(*, problem, model_config, method_config, optimizer_config, lr_schedule="constant",
          grad_clip_norm=None, device="cpu", seed=0, start_update=0, method_seed=None):
    config = method_config if isinstance(method_config, CONFIG_CLASS) else CONFIG_CLASS(**method_config)
    network = make_network("resnet_18", problem=problem, model_config=model_config, device=device, seed=seed)
    return CChainLearner(network=network, config=config, problem=problem,
                         optimizer_config=optimizer_config, lr_schedule=lr_schedule, grad_clip_norm=grad_clip_norm,
                         seed=seed if method_seed is None else method_seed, start_update=start_update)
