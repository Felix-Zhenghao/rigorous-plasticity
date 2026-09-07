import pytest
import torch

from testbed.core.consumption import Consumer
from testbed.data import DATASETS, tensor_bundle
from testbed.data.datasets import load_dataset
from testbed.training import make_paradigm
from testbed.training.class_incremental import ClassIncrementalConfig
from testbed.training.class_incremental.data import mixture_alpha, mixture_arrivals
from testbed.training.class_remap import ClassRemapConfig
from testbed.training.pixel_permutation import PixelPermutationConfig


def bundle(name="toy", *, channels=1, labels=(0, 1, 2, 3)):
    targets = torch.tensor(labels).repeat_interleave(10)
    inputs = torch.arange(len(targets) * channels * 16).float().reshape(-1, channels, 4, 4) / 1000
    return tensor_bundle(inputs, targets, inputs + 0.1, targets, name=name, output_ids=labels)


def paradigm(name="class_remap", data=None, **overrides):
    config = dict(dataset="toy", task_samples="pool", chunk_size="task", validation_fraction=0)
    config.update(stage_sizes=[2, 4]) if name == "class_incremental" else config.update(num_tasks=2)
    config.update(overrides)
    return make_paradigm(name, config, datasets=data or bundle(), seed=17)


def rows(data):
    return [data[i] for i in range(len(data))]


def by_id(data):
    return {int(source_id): (x, int(y)) for x, y, source_id in rows(data)}


def test_registered_metadata_and_stratified_split_identity():
    assert {"mnist", "fashion_mnist", "emnist_balanced", "cifar10", "cifar100", "svhn", "tiny_imagenet"} <= DATASETS.keys()
    p = paradigm(validation_fraction=0.2, pool_size=12)
    ids = p.metadata["split_ids"]
    assert len(ids["train"]) == 32 and len(ids["val"]) == 8 and len(ids["test"]) == 40
    assert not set(ids["train"].tolist()) & set(ids["val"].tolist())
    assert not set(ids["train"].tolist()) & set(ids["test"].tolist())
    assert set(p.get_data().data.ids.tolist()) <= set(ids["train"].tolist())
    assert len(p.pool) == 12
    synthetic = load_dataset("synthetic", seed=4, options={"n_train": 20, "n_test": 8, "num_classes": 4})
    assert synthetic.output_ids == (0, 1, 2, 3)
    assert synthetic.input_shape == (1, 8, 8)


def test_same_class_mapping_stable_labels_recurrence_and_source_ids():
    p = paradigm(num_tasks=4, recurrence_period=2, stable_classes=[0], first_mapping="identity")
    first = by_id(p.get_data().data)
    second_data = p.get_data().data
    second = by_id(second_data)
    assert first.keys() == second.keys()
    for source_id, (_, label) in first.items():
        assert second[source_id][1] == p.mappings[1][label]
    assert p.mappings[1][0] == 0
    heldout = p.get_eval_data("test")
    assert all(int(heldout[i][1]) == p.mappings[1][int(p.bundle.test.targets[i])] for i in range(len(heldout)))
    assert {key: value[1] for key, value in by_id(p.get_data().data).items()} == {key: value[1] for key, value in first.items()}
    assert len(set(second_data.ids.tolist())) == len(second_data)


def test_class_probability_allocation_and_no_silent_replacement():
    p = paradigm(task_samples=7, class_probs=[0.25] * 4, first_mapping="identity")
    labels = torch.tensor([y for _, y, _ in rows(p.get_data().data)])
    assert torch.bincount(labels, minlength=4).tolist() == [2, 2, 2, 1]
    with pytest.raises(ValueError, match="requires"):
        paradigm(task_samples=20, class_probs=[1, 0, 0, 0]).get_data()
    with pytest.raises(ValueError, match="without replacement"):
        paradigm(task_samples=41).get_data()
    repeated = paradigm(task_samples=100, sampling="with_replacement", class_probs=[1, 0, 0, 0], first_mapping="identity").get_data().data
    assert len(set(repeated.ids.tolist())) <= 10 and len(repeated) == 100
    assert all(y == 0 for _, y, _ in rows(repeated))


def test_pool_refresh_and_malformed_schedules():
    fixed = paradigm(pool_size=8)
    assert set(fixed.get_data().data.ids.tolist()) == set(fixed.get_data().data.ids.tolist())
    refreshed = paradigm(pool_size=8, pool_refresh="per_task")
    assert set(refreshed.get_data().data.ids.tolist()) != set(refreshed.get_data().data.ids.tolist())
    with pytest.raises(ValueError, match="exactly 2"):
        paradigm(task_samples=[4])
    with pytest.raises(ValueError, match="sum to one"):
        paradigm(class_probs=[1, 1, 1, 1])
    with pytest.raises(ValueError, match="Exactly one"):
        paradigm(epochs=1, updates=1)


@pytest.mark.parametrize("axes", ["spatial", "all_values"])
def test_pixel_train_eval_use_original_coordinates_and_identical_transform(axes):
    p = paradigm("pixel_permutation", data=bundle(channels=3), permutation_axes=axes)
    task = p.get_data().data
    source_index = int(task.indices[0])
    original = p.splits["train"][source_index][0]
    perm = p.permutations[0]
    expected = original.flatten()[perm].reshape_as(original) if axes == "all_values" else original.reshape(3, -1)[:, perm].reshape_as(original)
    assert torch.equal(task[0][0], expected)
    test = p.get_eval_data("test")
    original_test = p.splits["test"][0][0]
    expected_test = original_test.flatten()[perm].reshape_as(original_test) if axes == "all_values" else original_test.reshape(3, -1)[:, perm].reshape_as(original_test)
    assert torch.equal(test[0][0], expected_test)
    assert torch.equal(task[0][0], task[0][0]) and task[0][2] == task.ids[0]
    assert torch.equal(p.get_unseen_data("test")[0][0], original_test)


def test_partial_pixel_recurrence_and_eval_do_not_advance_rng():
    p = paradigm("pixel_permutation", num_tasks=4, recurrence_period=2, permuted_fraction=[0, 0.5, 0, 0.5])
    assert torch.equal(p.permutations[0], torch.arange(16))
    assert int((p.permutations[1] != torch.arange(16)).sum()) <= 8
    p.get_data()
    state, rng = p.state_dict(), torch.random.get_rng_state()
    p.get_eval_data("test")[0]
    assert torch.equal(rng, torch.random.get_rng_state()) and p.next_task == state["next_task"]
    p.get_data()
    assert torch.equal(p.get_data().data.permutation, p.permutations[0])
    with pytest.raises(ValueError, match="repeat"):
        paradigm("pixel_permutation", num_tasks=3, recurrence_period=2, permuted_fraction=[0, 1, 1])


def test_heterogeneous_schedules_keep_partial_chunks_and_distinct_clocks():
    p = paradigm("pixel_permutation", task_samples=[5, 7], chunk_size=[2, 3], epochs=[2, None], updates=[None, 3], shuffle_each_epoch=False)
    consumer = Consumer(p, batch_size=2, seed=23)
    sizes = []
    for batch in consumer:
        sizes.append(len(batch[0]))
        consumer.commit()
    assert len(sizes) == 2 * (1 + 1 + 1) + 3 * 3
    assert sizes[:6] == [2, 2, 2, 2, 1, 1]
    assert consumer.counters == {"arrivals": 12, "training_exposures": 23}


def test_class_incremental_cumulative_availability_and_fixed_canonical_outputs():
    p = paradigm("class_incremental", class_order=[3, 1, 2, 0])
    assert p.problem.output_ids == (0, 1, 2, 3)
    initial = p.get_data().data
    assert set(y for _, y, _ in rows(initial)) == {1, 3}
    assert set(y for _, y, _ in rows(p.get_eval_data("test"))) == {1, 3}
    expanded = p.get_data().data
    assert set(initial.ids.tolist()) <= set(expanded.ids.tolist())
    assert set(y for _, y, _ in rows(p.get_eval_data("test"))) == {0, 1, 2, 3}
    assert p.problem.output_ids == (0, 1, 2, 3)


def test_class_incremental_explicit_subset_defines_the_final_output_space():
    p = paradigm("class_incremental", stage_sizes=[1, 2], class_order=[3, 1])
    assert p.problem.output_ids == (1, 3)
    assert set(y for _, y, _ in rows(p.get_data().data)) == {3}
    assert set(y for _, y, _ in rows(p.get_data().data)) == {1, 3}


@pytest.mark.parametrize("order,fraction", [("iid", 1), ("class_ordered", 1), ("mixed", 0.5), ("mixed", 1)])
def test_example_expansion_constructs_disjoint_arrival_partitions(order, fraction):
    p = paradigm("class_incremental", progression="examples", stage_sizes=[0.5, 1.0], arrival_order=order,
                 uniform_fraction=fraction, class_order=[0, 1, 2, 3])
    first, second = p.stage_pools
    assert len(first) == 20 and len(second) == 40
    assert len(set(second.tolist())) == 40 and set(first.tolist()) <= set(second.tolist())
    if order == "class_ordered":
        assert set(p.sources[0]["train"].targets[first].tolist()) == {0, 1}
    assert len(p.get_eval_data("test")) == 40


def test_transfer_uses_only_destination_with_fixed_label_union():
    source, destination = bundle("source"), bundle("destination", labels=(0, 1))
    p = paradigm("class_incremental", data={"toy": source, "destination": destination}, progression="transfer",
                 target_dataset="destination", stage_sizes=["all", 0.5], target_label_map={0: 10, 1: 11})
    assert p.problem.output_ids == (0, 1, 2, 3, 10, 11)
    first = p.get_data().data
    second = p.get_data().data
    assert len(second) == 10 and not set(first.ids.tolist()) & set(second.ids.tolist())
    assert set(y for _, y, _ in rows(second)) <= {10, 11}
    assert set(p.get_eval_data("test").ids.tolist()) == set(destination.test.ids.tolist())
    assert p.problem.output_ids == (0, 1, 2, 3, 10, 11)


def test_mixture_probabilities_endpoints_and_zero_update_initial_stage():
    assert [mixture_alpha("linear", r, 3) for r in (1, 2, 3, 4)] == [0, 0.5, 1, 1]
    assert mixture_alpha("exponential", 4, 4, gamma=0.9) == pytest.approx(1 - 0.9**50)
    assert mixture_alpha("exponential", 5, 4, gamma=0.9) == 1
    draws = mixture_arrivals(torch.arange(10), torch.arange(20), 40000, 40000, "explicit", 1,
                             torch.Generator().manual_seed(4), values=[0.5])
    assert float((draws < 10).float().mean()) == pytest.approx(0.75, abs=0.01)
    p = paradigm("class_incremental", progression="examples", stage_sizes=[0.5, 1.0], task_samples=["pool", 12],
                 chunk_size=["task", 4], epochs=[None, 2], updates=[0, None], sampling="with_replacement",
                 transition=["abrupt", "linear"], transition_chunks=[1, 3])
    data = p.get_data().data
    assert p.active_task == 1 and p.next_task == 2 and len(p.metadata["tasks"]) == 1
    assert set(data.indices[:4].tolist()) <= set(p.stage_pools[0].tolist())
    assert torch.equal(data[0][0], data[0][0])
    assert p.get_data() is None


@pytest.mark.parametrize("name", ["class_remap", "pixel_permutation", "class_incremental"])
def test_data_state_restores_arrivals_targets_and_current_eval(name):
    p = paradigm(name, sampling="with_replacement", task_samples=13)
    p.get_data()
    state = p.state_dict()
    expected = rows(p.get_data().data)
    restored = paradigm(name, sampling="with_replacement", task_samples=13)
    restored.load_state_dict(state)
    actual = rows(restored.get_data().data)
    for left, right in zip(expected, actual):
        assert all(torch.equal(torch.as_tensor(a), torch.as_tensor(b)) for a, b in zip(left, right))
    assert [int(row[1]) for row in rows(p.get_eval_data("test"))] == [int(row[1]) for row in rows(restored.get_eval_data("test"))]


def test_s05_offsets_share_exact_inputs_centered_residuals_and_ids():
    def build(offset):
        return paradigm(num_tasks=1, first_mapping="identity", target_mode="fixed_regression", target_mean=offset,
                        target_scale=3.0, pool_size=16, epochs=None, updates=4)
    zero, shifted = build(0), build(8)
    a, b = zero.fixed_regression, shifted.fixed_regression
    assert zero.problem.loss_kind == "mse" and zero.problem.output_ids == (0,)
    assert torch.equal(a["inputs"], b["inputs"]) and torch.equal(a["example_ids"], b["example_ids"])
    assert torch.equal(a["residuals"], b["residuals"])
    assert b["residuals"].mean().abs() < 1e-6
    assert torch.allclose(b["targets"] - a["targets"], torch.full_like(a["targets"], 8))
    data = shifted.get_data().data
    assert torch.equal(data[0][0], b["inputs"][0]) and torch.equal(data[0][1], b["targets"][0])
    restored = build(8)
    restored.load_state_dict(shifted.state_dict())
    assert torch.equal(restored.fixed_regression["inputs"], b["inputs"])
    with pytest.raises(ValueError, match="complete fixed pool"):
        paradigm(num_tasks=1, first_mapping="identity", target_mode="fixed_regression", pool_size=16, task_samples=8)


def test_scientifically_invalid_incremental_modes_fail():
    with pytest.raises(ValueError, match="replacement"):
        paradigm("class_incremental", transition="linear", transition_chunks=2)
    with pytest.raises(ValueError, match="fewer arrival chunks"):
        paradigm("class_incremental", transition="linear", transition_chunks=2, sampling="with_replacement")
    with pytest.raises(ValueError, match="strictly increasing"):
        paradigm("class_incremental", stage_sizes=[4, 2])
    with pytest.raises(ValueError, match="updates must be positive"):
        paradigm("class_incremental", epochs=[1, None], updates=[None, 0])
    with pytest.raises(ValueError, match="only abrupt"):
        ClassIncrementalConfig(dataset="toy", progression="transfer", target_dataset="other", stage_sizes=["all", "all"],
                               task_samples="pool", chunk_size="task", transition="linear")


def test_dataset_stats_use_training_membership_only():
    data = tensor_bundle(torch.zeros(8, 1, 2, 2), torch.tensor([0, 1] * 4),
                         torch.full((2, 1, 2, 2), 100.0), torch.tensor([0, 1]))
    p = paradigm(data=data, normalization="dataset_stats", validation_fraction=0.25)
    assert torch.equal(p.metadata["preprocessing"]["mean"], torch.zeros(1, 1, 1))
    assert torch.isfinite(p.get_eval_data("test")[0][0]).all()


def test_config_dataclasses_are_independent():
    for cls in (ClassRemapConfig, PixelPermutationConfig, ClassIncrementalConfig):
        assert cls.__bases__ == (object,)


def test_teacher_regression_evaluation_keeps_the_training_target_function():
    data = bundle()
    data.test.data = data.train.data
    p = paradigm(data=data, num_tasks=1, first_mapping="identity", target_mode="fixed_regression",
                 target_family="teacher", teacher={"name": "mlp", "hidden_sizes": [8]}, pool_size=16, target_mean=8)
    artifact, evaluation = p.fixed_regression, p.get_eval_data("test")
    for source_id, target in zip(artifact["example_ids"], artifact["targets"]):
        raw_index = int(source_id) & 0xFFFFFFFF
        assert torch.allclose(evaluation[raw_index][1], target, atol=1e-6)


def test_retarget_state_preserves_source_and_changes_only_target_membership():
    from testbed.training.class_incremental.paradigm import retarget_state
    datasets = {"toy": bundle("branch_source"), "destination": bundle("branch_target")}
    kwargs = dict(data=datasets, progression="transfer", target_dataset="destination")
    original = paradigm("class_incremental", stage_sizes=["all", 0.25], **kwargs)
    original.get_data()
    source_state = original.state_dict()
    branch = paradigm("class_incremental", stage_sizes=["all", 0.5], **kwargs)
    branch.load_state_dict(retarget_state(source_state, branch))
    assert torch.equal(branch.stage_pools[0], original.stage_pools[0])
    assert len(original.stage_pools[1]) == 10 and len(branch.get_data().data) == 20
    assert len(source_state["metadata"]["tasks"]) == 1
    with pytest.raises(ValueError, match="already entered"):
        retarget_state(branch.state_dict(), original)


def test_synthetic_variants_have_distinct_stable_raw_ids():
    first = load_dataset("synthetic", seed=12)
    repeated = load_dataset("synthetic", seed=12)
    other_seed = load_dataset("synthetic", seed=13)
    other_shape = load_dataset("synthetic", seed=12, options={"input_shape": [1, 4, 4]})
    assert torch.equal(first.train.ids, repeated.train.ids)
    assert not set(first.train.ids.tolist()) & set(other_seed.train.ids.tolist())
    assert not set(first.train.ids.tolist()) & set(other_shape.train.ids.tolist())


def test_tiny_imagenet_uses_labeled_official_validation(tmp_path):
    from PIL import Image
    root = tmp_path / "tiny-imagenet-200"
    root.mkdir()
    (root / "wnids.txt").write_text("n2\nn1\n")
    for label in ("n1", "n2"):
        folder = root / "train" / label / "images"
        folder.mkdir(parents=True)
        Image.new("RGB", (4, 4), (12, 24, 36)).save(folder / "example.JPEG")
    (root / "val" / "images").mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(root / "val" / "images" / "val.JPEG")
    (root / "val" / "val_annotations.txt").write_text("val.JPEG\tn2\t0\t0\t4\t4\n")
    loaded = load_dataset("tiny_imagenet", tmp_path)
    assert loaded.test_designation == "official_labeled_validation"
    assert loaded.output_ids == (0, 1) and loaded.test[0][1] == 1
    assert loaded.train[0][0].shape == (3, 4, 4)


@pytest.mark.parametrize("name,count", [("mnist", 10), ("fashion_mnist", 10), ("emnist_balanced", 47),
                                         ("cifar10", 10), ("cifar100", 100), ("svhn", 10)])
def test_torchvision_adapters_keep_native_metadata_and_official_splits(monkeypatch, name, count):
    import sys
    from types import SimpleNamespace
    calls = []
    class Images:
        targets = labels = [0, 1]
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def __len__(self):
            return 2
        def __getitem__(self, index):
            return torch.zeros(1, 4, 4), self.targets[index]
    datasets = SimpleNamespace(**{key: Images for key in ("MNIST", "FashionMNIST", "EMNIST", "CIFAR10", "CIFAR100", "SVHN")})
    monkeypatch.setitem(sys.modules, "torchvision", SimpleNamespace(datasets=datasets))
    loaded = load_dataset(name, options={"download": False})
    assert len(loaded.output_ids) == count
    assert not set(loaded.train.ids.tolist()) & set(loaded.test.ids.tolist())
    if name == "svhn":
        assert [call["split"] for call in calls] == ["train", "test"]
    else:
        assert [call["train"] for call in calls] == [True, False]
    if name == "emnist_balanced":
        assert all(call["split"] == "balanced" for call in calls)


@pytest.mark.parametrize("override", [dict(target_family="teacher"), dict(teacher={"name": "mlp"}),
                                      dict(target_mean=8), dict(target_scale=2), dict(center_targets=False), dict(omega=10)])
def test_native_classification_rejects_inactive_regression_options(override):
    with pytest.raises(ValueError, match="require target_mode=fixed_regression"):
        ClassRemapConfig(dataset="toy", num_tasks=1, task_samples="pool", chunk_size="task", **override)


@pytest.mark.parametrize("override", [dict(teacher={"name": "mlp"}), dict(omega=10),
                                      dict(target_family="teacher", teacher={"name": "mlp"}, omega=10)])
def test_regression_rejects_options_ignored_by_its_target_family(override):
    with pytest.raises(ValueError, match="applies only"):
        ClassRemapConfig(dataset="toy", num_tasks=1, task_samples="pool", chunk_size="task",
                         target_mode="fixed_regression", first_mapping="identity", **override)


@pytest.mark.parametrize("progression", ["classes", "transfer"])
@pytest.mark.parametrize("override", [dict(arrival_order="class_ordered"), dict(arrival_order="mixed"), dict(uniform_fraction=0.5)])
def test_only_example_progression_accepts_example_arrival_options(progression, override):
    with pytest.raises(ValueError, match="require progression=examples"):
        ClassIncrementalConfig(dataset="toy", stage_sizes=[1, 2], task_samples="pool", chunk_size="task",
                               progression=progression, target_dataset="other" if progression == "transfer" else None, **override)


@pytest.mark.parametrize("override", [dict(target_data_options={"download": False}), dict(alpha_values=[None, [0, 1]]),
                                      dict(transition_gamma=0.9), dict(uniform_fraction=0.5),
                                      dict(transition=["exponential", "abrupt"], transition_gamma=0.9)])
def test_incremental_rejects_inactive_target_and_mixture_options(override):
    with pytest.raises(ValueError):
        ClassIncrementalConfig(dataset="toy", stage_sizes=[0.5, 1.0], task_samples="pool", chunk_size="task",
                               progression="examples", **override)


def test_first_stage_is_unmixed_and_explicit_alpha_entries_are_scoped():
    common = dict(dataset="toy", stage_sizes=[0.5, 1.0], task_samples=12, chunk_size=4, progression="examples",
                  sampling="with_replacement", transition_chunks=3)
    ClassIncrementalConfig(**common, transition="exponential", transition_gamma=0.9)
    ClassIncrementalConfig(**common, transition="explicit", alpha_values=[None, [0, 0.5, 1]])
    with pytest.raises(ValueError, match="must be null"):
        ClassIncrementalConfig(**common, transition="explicit", alpha_values=[[0, 0.5, 1], [0, 0.5, 1]])
