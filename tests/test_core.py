from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset

from testbed.core.config import TrainerConfig
from testbed.core.consumption import Consumer
from testbed.core.losses import supervised_loss
from testbed.core.optim import learning_rate
from testbed.core.types import Batch, Consumption, DataBlock, ProblemSpec


class Blocks:
    problem = ProblemSpec((1,), "mse", (0,))

    def __init__(
        self,
        lengths: tuple[int, ...] = (4,),
        consume: Consumption = Consumption(2, epochs=2, shuffle_each_epoch=False),
        stochastic: bool = False,
    ) -> None:
        self.lengths, self.consume, self.stochastic = lengths, consume, stochastic
        self.position = 0

    def get_data(self) -> DataBlock | None:
        if self.position == len(self.lengths):
            return None
        length = self.lengths[self.position]
        first = sum(self.lengths[:self.position])
        self.position += 1
        ids = torch.arange(first, first + length)
        data = TensorDataset(ids.float().unsqueeze(1), ids.float().unsqueeze(1), ids)
        return DataBlock(RandomAccess(data) if self.stochastic else data, self.consume)

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.position = state["position"]


class RandomAccess:
    def __init__(self, data: TensorDataset) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Batch:
        x, y, source_id = self.data[index]
        return x + torch.rand_like(x), y, source_id


def consume_all(consumer: Consumer) -> list[Batch]:
    result = []
    for batch in consumer:
        result.append(batch)
        consumer.commit()
    return result


def assert_state_equal(a: object, b: object) -> None:
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    elif hasattr(a, "shape"):
        assert (a == b).all()
    else:
        assert a == b


def test_exact_chunk_order_and_short_batches() -> None:
    consumer = Consumer(Blocks(), 1)
    assert [int(b[2]) for b in consume_all(consumer)] == [0, 1, 0, 1, 2, 3, 2, 3]
    assert consumer.counters == {"arrivals": 4, "training_exposures": 8}
    consume = Consumption(5, epochs=2, shuffle_each_epoch=False)
    batches = consume_all(Consumer(Blocks((7,), consume), 3))
    assert [b[2].tolist() for b in batches] == [[0, 1, 2], [3, 4], [0, 1, 2], [3, 4], [5, 6], [5, 6]]


def test_exact_update_budget_cycles_per_chunk() -> None:
    consumption = Consumption(3, epochs=None, updates=4, shuffle_each_epoch=False)
    consumer = Consumer(Blocks((5,), consumption), 2)
    assert [b[2].tolist() for b in consume_all(consumer)] == [[0, 1], [2], [0, 1], [2], [3, 4], [3, 4], [3, 4], [3, 4]]
    assert consumer.counters == {"arrivals": 5, "training_exposures": 14}


@pytest.mark.parametrize("workers", [0, 2])
def test_resume_committed_cursor_with_prefetch_and_augmentation(workers: int) -> None:
    consumption = Consumption(5, epochs=3)
    full = Consumer(Blocks((7, 4), consumption, True), 2, num_workers=workers, seed=41)
    trace = consume_all(full)
    interrupted = Consumer(Blocks((7, 4), consumption, True), 2, num_workers=workers, seed=41)
    for _ in range(4):
        next(interrupted)
        interrupted.commit()
    state = interrupted.state_dict()
    resumed = Consumer(Blocks((7, 4), consumption, True), 2, num_workers=workers, seed=41)
    resumed.load_state_dict(state)
    assert_state_equal(consume_all(resumed), trace[4:])
    assert resumed.counters == full.counters
    direct = Consumer(Blocks((7, 4), consumption, True), 2, num_workers=0, seed=41)
    assert_state_equal(trace, consume_all(direct))


def test_uncommitted_data_cannot_advance_or_checkpoint() -> None:
    consumer = Consumer(Blocks(), 1)
    next(consumer)
    with pytest.raises(RuntimeError, match="commit"):
        next(consumer)
    with pytest.raises(RuntimeError, match="commit"):
        consumer.state_dict()
    assert consumer.counters["arrivals"] == 0


def test_arrivals_count_available_chunk_separately_from_exposures() -> None:
    consumer = Consumer(Blocks((10,), Consumption(10, epochs=None, updates=1)), 2)
    consume_all(consumer)
    assert consumer.counters == {"arrivals": 10, "training_exposures": 2}


def test_singleton_online_creates_one_view_per_task() -> None:
    paradigm = Blocks((100,) * 1000, Consumption(1))
    # Exercise the entire lazy schedule without model allocation.
    from testbed.core.consumption import schedule
    count = sum(1 for _ in schedule(100, Consumption(1), 1, 0))
    assert count * len(paradigm.lengths) == 100_000
    consume_all(Consumer(Blocks((3,) * 10, Consumption(1)), 1))


def test_fixed_head_loss_and_no_regression_broadcasting() -> None:
    problem = ProblemSpec((1,), "cross_entropy", (4, 9, 20))
    logits = torch.tensor([[1., 2., 3.]], requires_grad=True)
    loss = supervised_loss(logits, torch.tensor([9]), problem)
    loss.backward()
    assert logits.grad[0, 2] > 0  # Unobserved output still participates in softmax.
    with pytest.raises(ValueError, match="broadcasting"):
        supervised_loss(torch.zeros(2, 1), torch.zeros(2), ProblemSpec((1,), "mse", (0,)))


def test_schedule_holds_terminal_and_validates_unused_fields() -> None:
    schedule = dict(name="cosine", horizon=4, terminal_lr=.02)
    assert learning_rate({"lr": .1}, schedule, 4) == .02
    assert learning_rate({"lr": .1}, schedule, 400) == .02
    with pytest.raises(ValueError):
        TrainerConfig(lr_schedule={"name": "constant", "gamma": .5})
    from testbed.core.optim import OptimizerConfig
    with pytest.raises(ValueError):
        OptimizerConfig(lr=float("nan"))
    with pytest.raises(ValueError):
        TrainerConfig(batch_size=True)
