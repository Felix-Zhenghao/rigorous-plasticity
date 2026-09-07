from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.LayerNorm):
    def __init__(self, shape, *, eps=1e-5, affine=True, gain="standard"):
        self.gain_mode = gain
        super().__init__(shape, eps=eps, elementwise_affine=affine)
        if affine and gain == "residual":
            nn.init.zeros_(self.weight)

    @property
    def gain(self):
        if self.weight is None:
            return None
        return self.weight + 1 if self.gain_mode == "residual" else self.weight

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.gain, self.bias, self.eps)


class ChannelLayerNorm(LayerNorm):
    def forward(self, x):
        x = super().forward(x.movedim(1, -1).contiguous())
        return x.movedim(-1, 1).contiguous()


def activation(name, negative_slope=0.01):
    return {"relu": lambda: nn.ReLU(), "leaky_relu": lambda: nn.LeakyReLU(negative_slope),
            "gelu": nn.GELU, "tanh": nn.Tanh}[name]()


def layer_norm(shape, config, *, channels=False):
    cls = ChannelLayerNorm if channels else LayerNorm
    return cls(shape, eps=config.ln_eps, affine=config.ln_affine, gain=config.ln_gain)


class HiddenLayer(nn.Module):
    def __init__(self, in_features, out_features, config):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=config.bias)
        self.norm = layer_norm(out_features, config) if config.norm == "layer" else nn.Identity()
        self.activation = activation(config.activation, config.negative_slope)
        self.dropout = nn.Dropout(getattr(config, "dropout", 0.0))
        self.norm_position = config.norm_position

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(self.norm(x)) if self.norm_position == "pre_activation" else self.norm(self.activation(x))
        return self.dropout(x)
