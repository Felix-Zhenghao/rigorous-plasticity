from testbed.core.factory import make_model

from .config import OptResetConfig
from .learner import OptimizerResetLearner


def build(*, problem, model_config, method_config, optimizer_config, lr_schedule="constant",
          grad_clip_norm=None, device="cpu", seed=0, start_update=0, method_seed=None):
    config = method_config if isinstance(method_config, OptResetConfig) else OptResetConfig(**method_config)
    base = make_model(config.base_method, "resnet_18", problem=problem, model_config=model_config,
                      method_config=config.base_config, optimizer_config=optimizer_config, lr_schedule=lr_schedule,
                      grad_clip_norm=grad_clip_norm, device=device, seed=seed, method_seed=method_seed, start_update=start_update)
    return OptimizerResetLearner(base, config)
