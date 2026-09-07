# Implementation validation

Validated on 2026-09-07 with Python 3.12.12, PyTorch 2.14.0+cu130, torchvision 0.29.0, and NVIDIA A100 80 GB GPUs. The local `.venv` contains the editable package and test dependencies.

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q` passes **165 tests**. Ruff's configured import and correctness checks also pass. Python 3.12 emits four deprecation warnings for the prefetched-loader test's use of multiprocessing fork from a multithreaded process; the test completes and reproduces its input trace exactly.

The suite checks all supported method/architecture pairs, penalties and source-operation fixtures, source IDs, class/pixel transformations, nested pools and mixtures, S05 target generation, optimizer moments and reset timing, exact checkpoint continuation, probe pairing/isolation, and sequential suites. The three stationary classification controls reach cross-entropy below 0.001; the tiny S05 regression control reaches MSE below 0.00001. These controls establish basic fitting ability, not empirical plasticity rankings.

GPU validation includes:

- C-CHAIN/MLP, CBP/ResNet-18, and FIRE/ViT through each of the three public training paradigms, with checkpoint/resume at update 4 and completion at update 8.
- A real torchvision MNIST permutation run with CBP/MLP, 20 completed updates, held-out evaluation, and checkpoints.
- Spectral/ViT, InFeR/ResNet, L2 Init/MLP, and channel-LayerNorm ResNet updates; both full-size ResNet and ViT example recipes construct successfully.
- Bitwise matching GPU dropout resume for weights, Adam statistics/counters, and RNG state; compatible probe optimizer carryover with the probe's own LR.
- Public training and teacher-probe CLI launches, and GPU training/resume with two CPU data-loader workers.

The CPU consumption benchmark exercised 1,000 tasks with 100 arrivals per task, using synthetic 1×8×8 images, replacement sampling, and no loader workers:

| Access regime | M | K | B | Updates | Arrivals | Exposures | Loading time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Online | 1 | 1 | 1 | 100,000 | 100,000 | 100,000 | 14.13 s |
| Repeated | 100 | 3 | 32 | 12,000 | 100,000 | 300,000 | 32.69 s |
| Intermediate | 10 | 2 | 4 | 60,000 | 100,000 | 200,000 | 25.16 s |

Each run used 1,000 task views. A separate 20-task A100 benchmark with a width-32 MLP reported loading/learning times of 0.41/2.49 s, 0.85/0.25 s, and 0.68/1.17 s respectively. GPU learner measurements synchronize the device. These measurements describe the small benchmark and machine; they do not estimate the plan's width-2000 research recipe.

Reproduce the measurements with `tools/benchmark_consumption.py`. Detailed integration artifacts are under `/net/spaces/scratch/zhenghao/tmp/testbed-final-gpu/`; benchmark JSON files are `/net/spaces/scratch/zhenghao/tmp/testbed-consumption-benchmark.json` and `/net/spaces/scratch/zhenghao/tmp/testbed-learner-benchmark.json`.

Long training sweeps and fresh-reference calibration for each intended research recipe remain empirical experiments. Author-operation agreement and implementation correctness do not establish that a method preserves plasticity in every setting.
