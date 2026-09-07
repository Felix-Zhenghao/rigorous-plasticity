"""Calibrate learnable stationary controls, without asserting plasticity loss."""
import pytest
import torch

from testbed.core.consumption import Consumer
from testbed.core.factory import make_model
from testbed.core.losses import supervised_loss
from testbed.data import tensor_bundle
from testbed.training import make_paradigm


def fit_objective(paradigm, *, hidden_sizes, batch_size, lr):
    learner = make_model("backprop", "mlp", problem=paradigm.problem,
                         model_config={"hidden_sizes": hidden_sizes},
                         optimizer_config={"name": "adam", "lr": lr}, seed=2, method_seed=7)
    consumer = Consumer(paradigm, batch_size=batch_size, seed=6)
    inputs, targets, ids = next(consumer)
    initial = supervised_loss(learner.predict(inputs), targets, paradigm.problem).item()
    learner.train_step((inputs, targets, ids))
    consumer.commit()
    for batch in consumer:
        learner.train_step(batch)
        consumer.commit()
    terminal = supervised_loss(learner.predict(inputs), targets, paradigm.problem).item()
    return initial, terminal, learner.completed_updates


@pytest.mark.parametrize("family", ["class_remap", "pixel_permutation", "class_incremental"])
def test_stationary_classification_objectives_are_learnable(family):
    config = dict(dataset="synthetic", task_samples="pool", chunk_size="task", epochs=None,
                  updates=100, validation_fraction=0,
                  data_options=dict(n_train=24, n_test=12, num_classes=3, input_shape=[1, 4, 4]))
    if family == "class_incremental":
        config.update(stage_sizes=[3], class_order=[2, 0, 1])
    else:
        config.update(num_tasks=1)
    paradigm = make_paradigm(family, config, seed=4)
    initial, terminal, updates = fit_objective(paradigm, hidden_sizes=[16], batch_size=24, lr=0.03)
    assert updates == 100
    assert terminal < 0.01 and terminal < initial / 100


def test_s05_fixed_iid_scalar_targets_are_learnable():
    # Independent input coordinates make this random finite target problem
    # representable by a scalar linear readout, including its nonzero mean.
    inputs = torch.eye(8).reshape(8, 1, 2, 4)
    labels = torch.arange(8) % 2
    data = tensor_bundle(inputs, labels, inputs, labels, name="orthogonal")
    config = dict(dataset="orthogonal", num_tasks=1, first_mapping="identity",
                  target_mode="fixed_regression", target_family="iid_normal",
                  target_mean=3.0, target_scale=0.5, validation_fraction=0,
                  task_samples="pool", chunk_size="task", epochs=None, updates=150)
    paradigm = make_paradigm("class_remap", config, seed=4, datasets=data)
    initial, terminal, updates = fit_objective(paradigm, hidden_sizes=[], batch_size=8, lr=0.05)
    assert updates == 150
    assert terminal < 1e-4 and terminal < initial / 1000
