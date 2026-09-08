# Configuration reference

Every configuration class has a field-by-field docstring immediately above its defaults. The comments describe the implemented behavior, including valid choices, units, and dependencies between options.

| YAML section | Field definitions and comments |
| --- | --- |
| Training recipe root; `trainer` | [`RunConfig`, `TrainerConfig`](../testbed/core/config.py) |
| `optimizer` (training or probes) | [`OptimizerConfig`](../testbed/core/optim.py) |
| `trainer.lr_schedule`; `probe.lr_schedule` | [`resolve_schedule`](../testbed/core/optim.py) |
| `model` | [`MLPConfig`, `ResNet18Config`, `ViTConfig`, presets](../testbed/models/config.py) |
| `data` for `class_remap` | [`ClassRemapConfig`](../testbed/training/class_remap/config.py) |
| `data` for `pixel_permutation` | [`PixelPermutationConfig`](../testbed/training/pixel_permutation/config.py) |
| `data` for `class_incremental` | [`ClassIncrementalConfig`](../testbed/training/class_incremental/config.py) |
| `data.data_options`; `data.target_data_options` | [Dataset loader options](../testbed/data/datasets.py) |
| Test recipe root | [`run_test`](../testbed/cli.py) |
| `probe` for `random_teacher` | [`TeacherProbeConfig`](../testbed/testing/random_teacher/config.py) |
| `probe` for `offset_refit` | [`OffsetProbeConfig`](../testbed/testing/offset_refit/config.py) |
| Suite recipe | [`run_suite`](../testbed/suite.py) |

For `method`, set `name` plus the fields of its config:

| Method | Config |
| --- | --- |
| `backprop` | [Backprop](../testbed/methods/backprop/config.py) |
| `l2` | [L2](../testbed/methods/l2/config.py) |
| `l2_init` | [L2 Init](../testbed/methods/l2_init/config.py) |
| `infer` | [InFeR](../testbed/methods/infer/config.py) |
| `feature_norm` | [Feature norm](../testbed/methods/feature_norm/config.py) |
| `c_chain` | [C-CHAIN](../testbed/methods/c_chain/config.py) |
| `spectral` | [Spectral](../testbed/methods/spectral/config.py) |
| `shrink_perturb` | [Shrink and perturb](../testbed/methods/shrink_perturb/config.py) |
| `cbp` | [CBP](../testbed/methods/cbp/config.py) |
| `redo` | [ReDo](../testbed/methods/redo/config.py) |
| `swr` | [SWR](../testbed/methods/swr/config.py) |
| `fire` | [FIRE](../testbed/methods/fire/config.py) |
| `layer_norm` | [LayerNorm](../testbed/methods/layer_norm/config.py) |
| `leaky_relu` | [Leaky ReLU](../testbed/methods/leaky_relu/config.py) |
| `nap` | [NaP](../testbed/methods/nap/config.py) |
| `optimizer_reset` | [Optimizer reset](../testbed/methods/optimizer_reset/config.py) |

The [supported method/architecture table](models_methods.md) lists valid pairs. Method constraints may change model settings; the run's saved `config.yaml` records the expanded settings actually used.

## Reading and changing a recipe

Omitting a field uses its declared default. YAML `null` becomes Python `None`; it only disables or derives a value where that field's comment says so. YAML lists supply sequence fields, including Python tuples. Unknown fields raise errors.

```bash
python run.py train --config configs/smoke.yaml \
  --set method.name=l2 --set method.coefficient=0.0001 \
  --set method.parameter_scope=head
```

In this example the added loss is `0.5 * 0.0001 * sum(theta**2)` for the selected parameters of the final output layer. A model's optional hidden head layers belong to the `hidden` scope. `include_bias`, `include_norm_affine`, and `include_embeddings` further filter a scope; the [L2 config](../testbed/methods/l2/config.py) explains exactly which parameters they include. Optimizer `weight_decay` is independent; enabling it and an L2 method applies both effects.

## Budget units

`task_samples` counts arriving positions in a task, and `chunk_size` limits how many are available together. `trainer.batch_size` limits examples per optimizer step. Repeated `epochs` or fixed `updates` apply **to each chunk**, not to the entire task. A final short minibatch remains within its pass and chunk.

For 100 arrivals with `chunk_size: 20`, `epochs: 3`, and `batch_size: 8`, there are five chunks and `5 * 3 * ceil(20 / 8) = 45` optimizer steps. Setting `epochs: null` and `updates: 10` instead gives `5 * 10 = 50` steps. `chunk_size: task` makes a whole task one chunk.

Schedules and reporting use the continuous completed-update clock; task changes do not restart it. Probe `updates` is the fitting budget for each branch in each repeat, starting from probe update zero. A suite's `repeat_seeds` and a probe's `repeats` are separate loops, so their counts multiply.

Run annotation and behavior checks with:

```bash
ruff check .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```
