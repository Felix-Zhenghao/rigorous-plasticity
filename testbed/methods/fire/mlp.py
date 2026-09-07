from torch import nn

from testbed.core.factory import make_network

from .config import CONFIG_CLASS
from .learner import FIRELearner


def build(*, problem, model_config, method_config, optimizer_config, lr_schedule="constant",
          grad_clip_norm=None, device="cpu", seed=0, start_update=0, method_seed=None):
    config = method_config if isinstance(method_config, CONFIG_CLASS) else CONFIG_CLASS(**method_config)
    network = make_network("mlp", problem=problem, model_config=model_config, device=device, seed=seed)
    selected = {f"{name}.weight": module.weight for name, module in network.named_modules()
                if isinstance(module, (nn.Linear, nn.Conv2d))}
    return FIRELearner(network=network, config=config, selected_weights=selected, problem=problem,
                         optimizer_config=optimizer_config, lr_schedule=lr_schedule, grad_clip_norm=grad_clip_norm,
                         seed=seed if method_seed is None else method_seed, start_update=start_update)
