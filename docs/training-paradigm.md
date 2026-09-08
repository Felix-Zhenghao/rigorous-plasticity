# supervised learning

## class remapping

### Dataset Classes

- **`RemappedData`** supplies selected images with remapped class labels for cross-entropy training (`target_mode="native"`). Within a task, images from the same original class receive the same emitted label, and different classes receive different labels. The mapping can change between tasks.
- **`FixedRegressionData`** stores input tensors and one fixed numerical target per image for MSE training (`target_mode="fixed_regression"`). Original class labels do not determine these targets. Repeated visits return the stored inputs and targets.

Both return `(x, target, source_id)`. `source_id` identifies the original dataset example and stays unchanged across remapping, augmentation, repeated draws, and tasks. These classes supply data for fitting; the training or probe procedure measures plasticity.

### Data Pipeline Overview

The classification pipeline is **dataset → eligible training pool → task arrivals → arrival chunks → minibatches → training updates**.

`dataset` selects the raw data, and `validation_fraction` reserves validation examples from its training split. `pool_size` then selects up to that many eligible training examples uniformly, without guaranteeing class proportions; `None` uses all remaining training examples. `pool_refresh="fixed"` keeps this pool across tasks, while `"per_task"` redraws it. For each task, `task_samples` draws are selected from the pool (`"pool"` means the pool's size). `sampling` controls replacement. `class_probs` applies to **original labels**: with replacement it controls random class draws; without replacement it determines rounded class quotas, which must fit the pool. When omitted, examples are sampled uniformly.

`first_mapping="identity"` preserves the first task's labels; `"random"` starts with a random permutation. `stable_classes` always keep their original labels, while the remaining classes permute among themselves; some may remain unchanged by chance. `recurrence_period=R` repeats a bank of R mappings cyclically; `None` draws a mapping for each task. Recurrence repeats label mappings, while pool and sample selection follow their own settings.

The selected arrivals are divided into **consecutive chunks** of `chunk_size` (`"task"` means all task arrivals). Each chunk gets either `epochs` complete passes or exactly `updates` optimizer steps, cycling through it as needed. `shuffle_each_epoch` controls reshuffling within that chunk on each pass. `trainer.batch_size` gives the usual minibatch size; fewer remaining examples produce a smaller batch. Final partial chunks and minibatches are retained, and batches never cross chunk or pass boundaries. For example, 10 arrivals with `chunk_size=6` and `batch_size=4` give chunks of 6 and 4, with batches `(4, 2)` and `(4)` per pass. An `updates` budget can stop mid-pass; even a short final chunk gets its full configured budget.

`num_tasks` sets the task count. `task_samples`, `chunk_size`, `epochs`, and `updates` can vary by task; `class_probs` can have one row per task. `resize` and `normalization` control input preprocessing; configured training augmentation is applied anew on each visit.

### MSE Target Computation

Fixed-regression mode uses one stationary task (`num_tasks=1`, `first_mapping="identity"`) and the entire fixed pool in one chunk (`pool_refresh="fixed"`, `task_samples="pool"`, `chunk_size="task"`, or equal numeric sizes). It requires `sampling="without_replacement"`, `class_probs=None`, and `augmentation="none"`.

First generate one base value `z_i` for each selected input `x_i`, according to `target_family`:

- **`iid_normal`**: draw an independent standard-normal value, `z_i ~ N(0, 1)`.
- **`teacher`**: use `z_i = teacher(x_i)`, where `teacher` specifies a randomly initialized, frozen network with one scalar output.
- **`sine_teacher`**: use `z_i = sin(omega * teacher(x_i))`; `omega` controls how rapidly the target oscillates with the teacher's output.

Then assign:

```text
c   = mean(z over the fixed training set) if center_targets else 0
y_i = target_mean + target_scale * (z_i - c)
```

Centering makes the training targets' mean equal to `target_mean`. `target_scale` multiplies the residuals without normalizing their variance; zero makes every target equal to `target_mean`. With centering disabled, `target_mean` is simply an additive offset. This offset stays constant during fitting. Defaults are `iid_normal`, centering enabled, scale 1, and offset 0.

Targets are generated once, stored alongside their inputs and IDs, and reused throughout fitting. Validation/test targets use the same training centering value: teacher families reuse the same teacher, while `iid_normal` draws separate fixed values for each split. The learner trains its scalar predictions against these targets using MSE.

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
