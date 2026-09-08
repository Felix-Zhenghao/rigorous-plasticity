# Supervised plasticity testbed

A PyTorch testbed for studying loss of plasticity under controlled changes in supervised learning. It implements [the design plan](testbed-impl-plan.md) with three training families, two isolated fitting assays, and backpropagation plus 15 maintenance methods.

Learners receive `(inputs, targets, source_ids)` and maintain one continuous optimizer-update clock. Data generation owns transformations and availability. The consumer controls chunk size, repeated passes, and minibatch size; task boundaries never reach learners or training metrics.

## Install and run

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[test]'
source .venv/bin/activate
python run.py train --config configs/smoke.yaml
python run.py train --config configs/pixel_online.yaml --set device=cuda:0
```

The smoke recipe uses deterministic synthetic images and needs no download. Real dataset recipes download through torchvision by default. Tiny ImageNet requires an extracted `tiny-imagenet-200` directory beneath `data_root`.

```bash
python run.py train --config configs/smoke.yaml \
  --set output_dir=runs/interrupted --stop-after-updates 5
python run.py train --resume runs/interrupted/checkpoints/update_5.pt
python run.py test --config configs/s03_teacher.yaml \
  --checkpoint runs/pixel_online/checkpoints/update_10000.pt
python run.py train --config configs/s05_pretrain.yaml
python run.py test --config configs/s05_offset.yaml \
  --checkpoint runs/s05_pretrain/checkpoints/final.pt
python run.py suite --config configs/supervised_suite.yaml
```

Use dotted `--set key=value` overrides; values follow YAML syntax. A method switch may require its own parameters, for example `--set method.name=l2_init --set method.coefficient=0.0001`. Unknown fields and unsupported method–architecture pairs raise errors. Existing training outputs require resume or a new output directory.

The [configuration guide](docs/configuration.md) links every configuration class and explains recipe structure, consumption budgets, and suite options. Each class documents its fields, accepted choices, and their effects next to the defaults.

## Experiments

| Family | Behavior | Recipes |
|---|---|---|
| S02 | Class remapping, stable classes, recurrence, fixed scalar-target pretraining | `s02_stream.yaml`, `s05_pretrain.yaml` |
| S07/S08 | Spatial/all-value pixel permutations, partial changes and recurrence | `pixel_online.yaml` |
| S12–S16 | Nested class/example pools, smooth mixtures, target-only transfer | `s12_class_incremental.yaml` through `s16_transfer.yaml` |
| S03 | Fixed same-architecture random-teacher targets on seen or unseen inputs | `s03_teacher.yaml` |
| S05 | Exact-input refits with independent residuals and paired target offsets | `s05_offset.yaml`, `offset_suite.yaml` |

For a task with `N` arrivals, `chunk_size=M`, `epochs=K`, and `trainer.batch_size=B`, updates equal `sum(K * ceil(chunk_length / B))`. `epochs: null` with `updates: U` instead allocates exactly U updates per chunk. Short minibatches stay within their pass and chunk. Replacement draws are fixed arrival positions, so replaying a position preserves its input identity and target. Training augmentation is seeded by logical visit independently of worker prefetching.

All configured output labels exist from initialization. Class expansion retains old examples; smooth transitions mix the old pool with the full expanded pool. S16 keeps the entire learner state and trains only on target inputs after source consumption. `transfer_suite.yaml` selects the source checkpoint by an explicit global update and branches target subset sizes from it.

All methods support MLP and ResNet-18: `backprop`, `l2`, `l2_init`, `infer`, `feature_norm`, `c_chain`, `spectral`, `shrink_perturb`, `cbp`, `redo`, `swr`, `fire`, `layer_norm`, `leaky_relu`, `nap`, and `optimizer_reset`. ViT supports the subset listed in [the model notes](docs/models_methods.md). Each method owns its configuration, update arithmetic, architecture-specific access, persistent state, and optimizer changes. The trainer contains no method dispatch during learning.

## Measurements and state

Each run saves its expanded YAML, software/architecture manifest, raw `metrics.jsonl`, initialization/final/update checkpoints, data memberships and transformations, and a bounded seen-input bank. Metrics distinguish pre-update loss/accuracy, cumulative online scores, current fitting scores, and held-out results. `arrivals` counts a chunk when it becomes available; `training_exposures` counts examples actually used by optimizer updates. All curves use completed updates.

The consumer commits data progress only after a successful learner update. Checkpoints restore network, optimizer, LR position, method histories/RNGs, and the next unconsumed minibatch, including when workers have prefetched ahead. Exact numeric equality is tested within the same software/device setup.

Probes keep the saved architecture, head, normalization buffers, and task loss. Maintenance penalties and interventions are inactive during fitting. They default to fresh Adam state at LR 0.001; `load_optimizer_state: true` restores compatible network statistics, excludes auxiliary heads, and still applies the probe LR and schedule. Aged/fresh branches share cached inputs, targets, fitting order, and budget. Probe artifacts are keyed by source checkpoint and assay fingerprints. Insufficient seen/unseen input banks raise errors.

Set `trainer.probe_at_updates` with `trainer.probes` for scheduled assays. Set `fresh_reference_at_updates` and `fresh_reference_updates` for paired windows on the next U training updates. Both leave primary training untouched. Checkpoint suites run one fitting branch at a time; experiment suites preserve raw test curves and aggregate final scores across independent seeds.

## Validation and provenance

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python tools/benchmark_consumption.py --tasks 1000 --samples 100
python tools/benchmark_consumption.py --tasks 20 --include-learner --device cuda:0
```

Tests cover data semantics, mathematical penalties, replacement masks and optimizer moments, method/architecture pairs, exact resume, probe isolation/pairing, and sequential suites. The benchmark reports loading and learning time separately for online, repeated, and intermediate access regimes.

The [validation record](docs/validation.md) documents the 165 passing checks, GPU integrations, stationary fitting controls, and measured consumption budgets.

Local papers are the primary scientific references. [Data notes](docs/data_notes.md), [model and regularizer notes](docs/models_methods.md), and [stateful-method notes](docs/stateful_methods.md) record source revisions and deliberate adaptations. Recipes with fixed heads, continuous schedules, or changed architecture/consumption budgets are testbed adaptations, not claims of exact paper reproduction. Long-run empirical plasticity comparisons and fresh-network budget calibration remain experiments to run with this testbed.
