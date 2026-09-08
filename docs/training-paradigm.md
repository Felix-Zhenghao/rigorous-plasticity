# supervised learning

## class remapping

### Dataset Classes

- **`RemappedData`** supplies selected images with remapped class labels for cross-entropy training (`target_mode="native"`). Within a task, images from the same original class receive the same emitted label, and different classes receive different labels. The mapping can change between tasks.
- **`FixedRegressionData`** stores input tensors and one numerical target per image for MSE training (`target_mode="fixed_regression"`). Original class labels do not determine these targets. Inputs stay fixed across tasks; targets stay fixed within each task and change with its teacher.

Both return `(x, target, source_id)`. `source_id` identifies the original dataset example and stays unchanged across remapping, augmentation, repeated draws, and tasks. These classes supply data for fitting; the training or probe procedure measures plasticity.

### Data Pipeline Overview

The classification pipeline is **dataset → eligible training pool → task arrivals → arrival chunks → minibatches → training updates**.

`dataset` selects the raw data, and `validation_fraction` reserves validation examples from its training split. `pool_size` then selects up to that many eligible training examples uniformly, without guaranteeing class proportions; `None` uses all remaining training examples. `pool_refresh="fixed"` keeps this pool across tasks, while `"per_task"` redraws it. For each task, `task_samples` draws are selected from the pool (`"pool"` means the pool's size). `sampling` controls replacement. `class_probs` applies to **original labels**: with replacement it controls random class draws; without replacement it determines rounded class quotas, which must fit the pool. When omitted, examples are sampled uniformly.

`first_mapping="identity"` preserves the first task's labels; `"random"` starts with a random permutation. `stable_classes` always keep their original labels, while the remaining classes permute among themselves; some may remain unchanged by chance. `recurrence_period=R` repeats a bank of R mappings cyclically; `None` draws a mapping for each task. Recurrence repeats label mappings, while pool and sample selection follow their own settings.

The selected arrivals are divided into **consecutive chunks** of `chunk_size` (`"task"` means all task arrivals). Each chunk gets either `epochs` complete passes or exactly `updates` optimizer steps, cycling through it as needed. `shuffle_each_epoch` controls reshuffling within that chunk on each pass. `trainer.batch_size` gives the usual minibatch size; fewer remaining examples produce a smaller batch. Final partial chunks and minibatches are retained, and batches never cross chunk or pass boundaries. For example, 10 arrivals with `chunk_size=6` and `batch_size=4` give chunks of 6 and 4, with batches `(4, 2)` and `(4)` per pass. An `updates` budget can stop mid-pass; even a short final chunk gets its full configured budget.

`num_tasks` sets the task count. `task_samples`, `chunk_size`, `epochs`, and `updates` can vary by task; `class_probs` can have one row per task. `resize` and `normalization` control input preprocessing; configured training augmentation is applied anew on each visit.

### MSE Target Computation

Fixed-regression mode fits the same input tensors across `num_tasks` tasks. At each task boundary, it independently reinitializes the teacher's weights and generates a new target set. Each task uses the entire fixed pool in one chunk (`pool_refresh="fixed"`, `task_samples="pool"`, `chunk_size="task"`, or equal numeric sizes). It requires `first_mapping="identity"`, `sampling="without_replacement"`, `class_probs=None`, and `augmentation="none"`. A single task remains a stationary fitting control.

For each task, generate one base value `z_i` for each selected input `x_i`, according to `target_family`:

- **`teacher`**: use `z_i = teacher(x_i)`, where `teacher` specifies a randomly initialized, frozen network with one scalar output.
- **`sine_teacher`**: use `z_i = sin(omega * teacher(x_i))`; `omega` controls how rapidly the target oscillates with the teacher's output.

Then assign:

```text
c   = mean(z over the fixed training set) if center_targets else 0
y_i = target_mean + target_scale * (z_i - c)
```

Centering makes each task's training targets' mean equal to `target_mean`. `target_scale` multiplies the residuals without normalizing their variance; zero makes every target equal to `target_mean`. With centering disabled, `target_mean` is simply an additive offset. Defaults are `teacher`, centering enabled, scale 1, and offset 0. Both target families require a `teacher` architecture recipe.

`target_mean`, `target_scale`, and `omega` each accept a scalar for all tasks or a list of exactly `num_tasks` values. Scheduled values change only at task boundaries; `omega` applies only to `sine_teacher`. For example:

```yaml
data:
  dataset: mnist
  num_tasks: 3
  first_mapping: identity
  pool_size: 1000
  task_samples: pool
  chunk_size: task
  target_mode: fixed_regression
  target_family: sine_teacher
  teacher: {name: mlp, hidden_sizes: [100, 100]}
  target_mean: [0, 8, 16]
  target_scale: [1, 1, 2]
  omega: [100000, 10000, 1000]
```

Targets are cached within each task and keyed by unchanged source IDs. Validation and test use the active task's teacher, target parameters, and training centering value. Checkpoints save that task's exact inputs, targets, residuals, resolved parameters, and teacher seed, so evaluation, resume, and offset refits use the same target function. Teacher seeds are independent of target offsets, scales, and frequencies. Class-mapping recurrence does not repeat teachers.

Implementation: [configuration](rigorous-plasticity/testbed/training/class_remap/config.py), [dataset classes and target generators](rigorous-plasticity/testbed/training/class_remap/data.py), [task construction](rigorous-plasticity/testbed/training/class_remap/paradigm.py), and [chunk/minibatch scheduling](rigorous-plasticity/testbed/core/consumption.py).

## pixel permutation

### Data Pipeline Overview

The classification pipeline is **dataset → eligible training pool → task arrivals → arrival chunks → minibatches → training updates**.

`dataset` selects the raw data, and `validation_fraction` reserves validation examples from its training split. `pool_size` then selects up to that many eligible training examples uniformly, without guaranteeing class proportions; `None` uses all remaining training examples. `pool_refresh="fixed"` keeps this pool across tasks, while `"per_task"` redraws it. For each task, `task_samples` draws are selected from the pool (`"pool"` means the pool's size). `sampling` controls replacement. `class_probs` applies to **original labels**: with replacement it controls random class draws; without replacement it determines rounded class quotas, which must fit the pool. When omitted, examples are sampled uniformly.

`PermutedData` applies one **fixed pixel permutation per task**, preserving each image's original class label and `source_id`. Training, validation, and test images for that task share the same permutation, and training uses cross-entropy. `permutation_axes="spatial"` rearranges the H×W positions identically in every channel; `"all_values"` rearranges all C×H×W values and can mix channels. `permuted_fraction` selects `floor(fraction * position_count)` random positions to permute among themselves; other positions stay fixed, and some selected positions may also remain unchanged by chance.

`first_permutation="identity"` applies no pixel permutation in the first task, regardless of `permuted_fraction`; `"random"` starts with a random permutation using the configured fraction. `recurrence_period=R` repeats a bank of R permutations cyclically; `None` draws a permutation for each task. Recurrence repeats pixel permutations, while pool and sample selection follow their own settings.

The selected arrivals are divided into **consecutive chunks** of `chunk_size` (`"task"` means all task arrivals). Each chunk gets either `epochs` complete passes or exactly `updates` optimizer steps, cycling through it as needed. `shuffle_each_epoch` controls reshuffling of examples within that chunk on each pass; the pixel permutation stays fixed. `trainer.batch_size` gives the usual minibatch size; fewer remaining examples produce a smaller batch. Final partial chunks and minibatches are retained, and batches never cross chunk or pass boundaries. For example, 10 arrivals with `chunk_size=6` and `batch_size=4` give chunks of 6 and 4, with batches `(4, 2)` and `(4)` per pass. An `updates` budget can stop mid-pass; even a short final chunk gets its full configured budget.

`num_tasks` sets the task count. `task_samples`, `chunk_size`, `epochs`, `updates`, and `permuted_fraction` can vary by task; `class_probs` can have one row per task. With recurrence enabled, the `permuted_fraction` schedule must repeat with the permutation bank. Inputs are resized according to `resize`, augmented if configured, then permuted. `normalization="dataset_stats"` applies channel standardization afterward. Training augmentation is applied anew on each visit; validation and test use no augmentation.

Implementation: [configuration](rigorous-plasticity/testbed/training/pixel_permutation/config.py), [PermutedData](rigorous-plasticity/testbed/training/pixel_permutation/data.py), [task construction](rigorous-plasticity/testbed/training/pixel_permutation/paradigm.py), [input preprocessing](rigorous-plasticity/testbed/data/datasets.py), and [chunk/minibatch scheduling](rigorous-plasticity/testbed/core/consumption.py).

## class incremental learning

Class incremental training uses real datasets for both source and transfer target. `progression="classes"` expands the available class set, `"examples"` expands a nested example pool, and `"transfer"` switches to target-only data. The complete configured output space exists from initialization.

For class expansion, `class_probs` is masked to classes present in the current eligible pool and renormalized for each stage. Supply one vector in sorted final output-ID order or one row per stage. At least one available class must have positive probability; without-replacement class quotas must fit the pool.

For example expansion, `arrival_order="iid"` takes prefixes of a random pool order; `"class_ordered"` groups the pool by `class_order` before taking prefixes. These are the two supported arrival orders.

Smooth transitions mix the old pool with the expanded pool using `linear`, `exponential`, or `explicit` coefficients and require `class_probs=None`. Both replacement policies are supported. Without replacement, `task_samples` must not exceed the expanded pool size, and every draw removes the selected example from both pools for the rest of the stage. Once the old pool is exhausted, sampling uses only the remaining expanded pool regardless of the coefficient. Each sampled arrival stays fixed during repeated fitting.
