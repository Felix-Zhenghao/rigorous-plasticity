from dataclasses import asdict


class OptimizerResetLearner:
    def __init__(self, base, config):
        self.base, self.config = base, config
        if getattr(base, "config", None) is not None and getattr(base.config, "optimizer_state", None) == "reset_all":
            raise ValueError("base method already resets all optimizer state")
        self.resets = 0
        self.resolved_model_config = base.resolved_model_config
        self.resolved_method_config = asdict(config)
        self.selected_parameters = base.selected_parameters

    @property
    def rng(self):
        return self.base.rng

    @rng.setter
    def rng(self, value):
        self.base.rng = value

    @property
    def optimizer_config(self):
        return self.base.optimizer_config

    @property
    def lr_schedule(self):
        return self.base.lr_schedule

    @property
    def problem(self):
        return self.base.problem

    @property
    def device(self):
        return self.base.device

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def network(self):
        return self.base.network

    @property
    def optimizer(self):
        return self.base.optimizer

    @property
    def completed_updates(self):
        return self.base.completed_updates

    @completed_updates.setter
    def completed_updates(self, value):
        self.base.completed_updates = value

    def predict(self, x):
        return self.base.predict(x)

    def train_step(self, batch):
        result = self.base.train_step(batch)
        due = (self.completed_updates in self.config.at_updates or self.config.every_updates is not None
               and self.completed_updates % self.config.every_updates == 0)
        if due:
            self.optimizer.state.clear()
            self.resets += 1
        result.metrics["optimizer_reset"] = float(due)
        return result

    @property
    def cost_metrics(self):
        return self.base.cost_metrics

    def state_dict(self):
        state = self.base.state_dict()
        state["optimizer_reset"] = {"resets": self.resets, "config": asdict(self.config)}
        state["resolved_method_config"] = self.resolved_method_config
        return state

    def load_state_dict(self, state):
        if state["optimizer_reset"]["config"] != asdict(self.config):
            raise ValueError("optimizer reset configuration differs from checkpoint")
        self.base.load_state_dict(state)
        self.resets = state["optimizer_reset"]["resets"]
