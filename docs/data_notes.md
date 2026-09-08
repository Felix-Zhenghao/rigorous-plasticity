# Data protocols and provenance

The data implementations follow `testbed-impl-plan.md`. They implement experiments, not claims that any one recipe reproduces a paper's results. Every task view contains source positions and its fixed label mapping or coordinate permutation. Repeated access preserves source identity and targets. Augmentation is a separate operation, with its random seed assigned by the consumer to each logical visit. Decoding/resizing precedes augmentation, then permutation, then normalization.

## Dataset identity and membership

The adapters use torchvision's official MNIST, Fashion-MNIST, EMNIST-balanced, CIFAR-10, CIFAR-100, and SVHN splits. EMNIST-balanced has **47 classes**. Tiny ImageNet reads the official extracted `tiny-imagenet-200` layout; its labeled validation images are explicitly designated as the reporting split because the challenge test set has no public labels. Download/extraction of Tiny ImageNet is manual. Torchvision adapters accept `data_options: {download: false}` for local-only loading. Class incremental training requires real source and target datasets. The synthetic adapter supplies deterministic prototype-plus-noise images for offline class-remapping and pixel-permutation checks.

Source IDs are signed 64-bit integers: a 31-bit BLAKE2 namespace derived from dataset name/version/raw split, followed by the original 32-bit index. They contain no task, transformation, chunk, or pass information. A validation example retains its original training-split source identity; it is removed from the eligible training membership. Namespace collisions fail at loading, and actual split-ID tensors are saved and checked on restore. Synthetic identity includes the generation seed and all shape/size/class options. Custom tensor adapters require a distinct name/version for distinct underlying data. The preprocessing identity and label maps are recorded separately from source IDs.

Validation membership is selected once, stratified by native class, with `floor(class_count * validation_fraction)` examples per class. Pool sizes cap the resulting training pool. `dataset_stats` computes per-channel statistics from the retained training membership, after deterministic resizing, and never inspects held-out values. Actual training/validation/test memberships, pools, mappings or permutations, and realized class proportions are saved in data state. Arrival identities are reconstructed from purpose-separated deterministic seeds, avoiding an accumulating copy of every past arrival in every checkpoint.

When class probabilities are specified, without-replacement sampling uses largest remainders with canonical-label tie breaking and fails on class shortages. Replacement sampling draws a class then a uniformly selected member of that class. Empirical sampling draws uniformly from eligible examples. Input pools and transformation banks use independent random streams.

## Class mappings and coordinate permutations

S02 maps original labels to new labels consistently in training and held-out views. Stable classes remain identity mapped; complementary labels undergo an ordinary uniform permutation, which may contain fixed points. Recurrence repeats the mapping bank, while sampling budgets and arrival IDs can change.

S07/S08 apply each permutation to original coordinates, including in evaluation. `spatial` shares one H×W permutation across channels; `all_values` can mix channels. A partial permutation selects `floor(fraction * coordinate_count)` eligible positions once. Actual moved fractions may be smaller. These RGB policies, partial permutations, recurrence, heterogeneous schedules, and holdouts are explicit testbed choices. `pixel_online.yaml` is the plan's illustrative 1,000 × 100 online stream. P09's main reported MNIST stream instead uses 800 × 60,000 arrivals.

## Incremental pools and smoothing

Class/example expansion retains every previously introduced example. The output head contains the complete final configured output space from initialization, in canonical order, and all outputs participate in every softmax. This intentionally differs from P09's growing output head. Learning rates, optimizer statistics, and maintenance histories continue across every stage; paper procedures with stagewise restart/reset are not silently reproduced.

For `progression="classes"`, supplied `class_probs` are masked to the classes present in each stage's eligible pool and renormalized before drawing arrivals. The same probability vector can therefore be reused as classes become available; a per-stage matrix is also supported. At least one available class must retain positive probability. Without-replacement class quotas must still fit the eligible pool.

Example arrival supports `iid` and `class_ordered`. IID arrival takes nested prefixes of a seeded random pool order; class-ordered arrival groups that pool by the configured class order before taking prefixes. Stage sizes define cumulative totals, and each example is introduced only once.

Smooth sampling uses the mixture `(1-alpha) Uniform(A) + alpha Uniform(D)`, where the expanded pool D includes A, and requires `class_probs=None`. Thus old membership has probability `1-alpha + alpha*|A|/|D|`. With replacement, the component is chosen independently for every arrival, as in P02.

Without replacement, `task_samples` must not exceed the expanded pool size. The mixture uses the remaining examples in A and D after each draw, and each selected example is removed from both pools. If A becomes empty, all subsequent arrivals come from the remaining D, which then contains only newly added examples, even when alpha is below one. Arrivals are unique across the entire stage, including across chunks; later stages may reuse earlier examples.

Arrival chunk r is one based. Linear ramps begin at zero and reach one at R; exponential ramps use the paper's `1-gamma**(50*r/R)` and explicitly become one after R. The last exponential ramp chunk need not reach one. Repeated chunk fitting does not advance the ramp. The optional zero-update initial stage establishes A without producing training arrivals.

S16 switches to target-only data and target held-out reporting. The native default CIFAR-10/SVHN IDs intentionally reuse ten output units despite different semantics. Explicit maps can instead establish disjoint output meanings. Source and target must produce identical input shapes. `retarget_state` permits target-size branches while preserving source membership and the full label mapping; the suite also validates the exact completed source update.

## Fixed-input regression and S05 refits

The S02 regression adapter holds preprocessed inputs and source IDs fixed across tasks. Teacher and sine-teacher targets use a frozen scalar network built through the ordinary architecture factory, with independently reinitialized weights at every task boundary. Targets are `a + target_scale * (base - mean(base))` when centering is enabled. Mean, scale, and sine frequency each accept a scalar or one value per task; centering is recomputed from that task's training targets and shared with held-out evaluation. These parameters never participate in input, teacher, or residual seeds.

Checkpoints save the active task's exact inputs, source IDs, residuals, targets, resolved offset/scale/frequency, and teacher definition/seed. S05 refits draw a new teacher on the same saved X, using that task's target family and resolved scale/frequency; `"same"` selects its resolved offset. The single-task `s05_pretrain.yaml` recipe remains a stationary control.

The source motivation is P10, _Disentangling the Causes of Plasticity Loss in Neural Networks_, §3.1/Fig. 1 and Appendix F.1. Its CIFAR-10 appendix setup (10,000 inputs, sine frequency 10⁵, B=512, Adam LR=0.001, offsets 0/8/16/32) is distinct from the smaller MNIST teacher recipe. Finite-set centering and task-dependent teacher targets and parameters are controlled extensions.

## Validation

Configuration rejects nondefault options that cannot affect the selected target family, progression, or transition. Regression-only fields require fixed regression; frequency overrides require sine-teacher targets; example-arrival options require example progression. Parameter schedules must have exactly one entry per task. Explicit coefficient lists use null entries for the first stage and every stage without an explicit transition. Default placeholder values remain valid in resolved recipes.

`tests/test_data.py` checks label consistency, train/evaluation permutations, recurrence, split/pool exclusion, exact class allocation, replacement policy, heterogeneous consumption counts, nested availability, transfer-only sampling, mixture endpoints and membership probabilities, zero-update initial stages, deterministic restore, and exact-X S05 offset pairing. `tests/test_regression_tasks.py` checks changing teachers, target schedules, held-out centering, and restored task state. These checks establish protocol correctness; they do not assert that training loses or retains plasticity.
