# Data protocols and provenance

The data implementations follow `testbed-impl-plan.md`. They implement experiments, not claims that any one recipe reproduces a paper's results. Every task view contains source positions and its fixed label mapping or coordinate permutation. Repeated access preserves source identity and targets. Augmentation is a separate operation, with its random seed assigned by the consumer to each logical visit. Decoding/resizing precedes augmentation, then permutation, then normalization.

## Dataset identity and membership

The adapters use torchvision's official MNIST, Fashion-MNIST, EMNIST-balanced, CIFAR-10, CIFAR-100, and SVHN splits. EMNIST-balanced has **47 classes**. Tiny ImageNet reads the official extracted `tiny-imagenet-200` layout; its labeled validation images are explicitly designated as the reporting split because the challenge test set has no public labels. Download/extraction of Tiny ImageNet is manual. Torchvision adapters accept `data_options: {download: false}` for local-only loading. The synthetic adapter is a deterministic, learnable prototype-plus-noise dataset for offline checks, not a research benchmark.

Source IDs are signed 64-bit integers: a 31-bit BLAKE2 namespace derived from dataset name/version/raw split, followed by the original 32-bit index. They contain no task, transformation, chunk, or pass information. A validation example retains its original training-split source identity; it is removed from the eligible training membership. Namespace collisions fail at loading, and actual split-ID tensors are saved and checked on restore. Synthetic identity includes the generation seed and all shape/size/class options. Custom tensor adapters require a distinct name/version for distinct underlying data. The preprocessing identity and label maps are recorded separately from source IDs.

Validation membership is selected once, stratified by native class, with `floor(class_count * validation_fraction)` examples per class. Pool sizes cap the resulting training pool. `dataset_stats` computes per-channel statistics from the retained training membership, after deterministic resizing, and never inspects held-out values. Actual training/validation/test memberships, pools, mappings or permutations, and realized class proportions are saved in data state. Arrival identities are reconstructed from purpose-separated deterministic seeds, avoiding an accumulating copy of every past arrival in every checkpoint.

When class probabilities are specified, without-replacement sampling uses largest remainders with canonical-label tie breaking and fails on class shortages. Replacement sampling draws a class then a uniformly selected member of that class. Empirical sampling draws uniformly from eligible examples. Input pools and transformation banks use independent random streams.

## Class mappings and coordinate permutations

S02 maps original labels to new labels consistently in training and held-out views. Stable classes remain identity mapped; complementary labels undergo an ordinary uniform permutation, which may contain fixed points. Recurrence repeats the mapping bank, while sampling budgets and arrival IDs can change.

S07/S08 apply each permutation to original coordinates, including in evaluation. `spatial` shares one H×W permutation across channels; `all_values` can mix channels. A partial permutation selects `floor(fraction * coordinate_count)` eligible positions once. Actual moved fractions may be smaller. These RGB policies, partial permutations, recurrence, heterogeneous schedules, and holdouts are explicit testbed choices. `pixel_online.yaml` is the plan's illustrative 1,000 × 100 online stream. P09's main reported MNIST stream instead uses 800 × 60,000 arrivals.

## Incremental pools and smoothing

Class/example expansion retains every previously introduced example. The output head contains the complete final configured output space from initialization, in canonical order, and all outputs participate in every softmax. This intentionally differs from P09's growing output head. Learning rates, optimizer statistics, and maintenance histories continue across every stage; paper procedures with stagewise restart/reset are not silently reproduced.

For mixed example arrival, the source of truth inspected was **P02, _A Study on the Plasticity of Neural Networks_, Appendix A.3**, in `papers/`. Its construction partitions classes into stage groups, randomly splits the full pool into disjoint IID and class components, partitions the IID component, and accumulates both components. Here class groups are contiguous balanced partitions of the configured class order. The number of IID examples needed at each stage is determined by the requested exact total and that stage's class component. Incompatible totals fail explicitly. Unequal stage totals and the exact rounding rule are declared extensions; no example is introduced twice. A final unrequested remainder partition is included when the last configured availability is less than the whole pool.

Smooth sampling implements P02's mixture `(1-alpha) Uniform(A) + alpha Uniform(D)`, where the expanded pool D includes A. Thus old membership has probability `1-alpha + alpha*|A|/|D|`. The component is chosen independently for every arrival, with replacement. Arrival chunk r is one based. Linear ramps begin at zero and reach one at R; exponential ramps use the paper's `1-gamma**(50*r/R)` and explicitly become one after R. The last exponential ramp chunk need not reach one. Repeated chunk fitting does not advance the ramp. The optional zero-update initial stage establishes A without producing training arrivals.

S16 switches to target-only data and target held-out reporting. The native default CIFAR-10/SVHN IDs intentionally reuse ten output units despite different semantics. Explicit maps can instead establish disjoint output meanings. Source and target must produce identical input shapes. `retarget_state` permits target-size branches while preserving source membership and the full label mapping; the suite also validates the exact completed source update.

## S05 fixed regression

The stationary S02 adapter saves the exact preprocessed input tensors, source IDs, residuals, offsets, and generator definition. Targets are `a + target_scale * (base - mean(base))` when centering is enabled. Offsets never participate in input, teacher, or residual seeds. IID-normal targets are a controlled testbed option. Teacher and sine-teacher families construct an independently initialized, frozen scalar network through the ordinary architecture factory. Refits generate new residuals on the same saved X.

The source motivation is P10, _Disentangling the Causes of Plasticity Loss in Neural Networks_, §3.1/Fig. 1 and Appendix F.1. Its CIFAR-10 appendix setup (10,000 inputs, sine frequency 10⁵, B=512, Adam LR=0.001, offsets 0/8/16/32) is distinct from the small IID-normal recipe. Finite-set centering and the configurable target families are controlled extensions. No author implementation was needed to define raw dataset transformations; the mixed-partition construction was checked directly in the local paper.

## Validation

Configuration rejects nondefault options that cannot affect the selected target family, progression, or transition. Regression-only fields require fixed regression; example-arrival options require example progression; `uniform_fraction` requires mixed arrival. Explicit coefficient lists use null entries for the first stage and every stage without an explicit transition. Default placeholder values remain valid in resolved recipes.

`tests/test_data.py` checks label consistency, train/evaluation permutations, recurrence, split/pool exclusion, exact class allocation, replacement policy, heterogeneous consumption counts, nested availability, transfer-only sampling, mixture endpoints and membership probabilities, zero-update initial stages, deterministic restore, and exact-X S05 offset pairing. These checks establish protocol correctness; they do not assert that training loses or retains plasticity.
