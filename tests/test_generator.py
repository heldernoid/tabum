"""Generator smoke + coverage tests (fast CPU checks, not the Phase 1 gate —
the distributional validation against real datasets is a separate, human-
reviewed step)."""

import numpy as np
import pytest

from tabum.generator import GeneratorConfig, TaskSampler


@pytest.fixture(scope="module")
def tasks():
    sampler = TaskSampler(GeneratorConfig(max_rows=512, max_features=30), seed=1)
    return [sampler.sample() for _ in range(60)]


def test_basic_validity(tasks):
    for t in tasks:
        assert t.X.dtype == np.float32
        assert t.X.shape[0] == t.y.shape[0]
        assert t.X.shape[1] >= 2
        assert 0 < t.train_size < t.X.shape[0]
        finite = t.X[~np.isnan(t.X)]
        assert np.isfinite(finite).all()
        if t.task_type == "classification":
            assert t.y.dtype == np.int64
            assert t.n_classes >= 2
            train_classes = np.unique(t.y[: t.train_size])
            assert train_classes.size == t.n_classes  # every class seen in train
            assert np.isin(np.unique(t.y), train_classes).all()  # no unseen test class
        else:
            assert t.y.dtype == np.float32
            assert np.isfinite(t.y).all()


def test_family_and_type_coverage(tasks):
    families = {t.family for t in tasks}
    types = {t.task_type for t in tasks}
    assert families == {"scm", "gp", "tree"}
    assert types == {"classification", "regression"}
    assert any(t.shift_split for t in tasks)


def test_quirk_coverage(tasks):
    assert any(np.isnan(t.X).any() for t in tasks), "no missingness generated"
    assert any(t.cat_mask.any() for t in tasks), "no categorical columns generated"
    # scale diversity: some column somewhere should have a large dynamic range
    max_scale = max(
        np.nanstd(t.X[:, j]) for t in tasks for j in range(t.X.shape[1])
    )
    min_scale = min(
        np.nanstd(t.X[:, j])
        for t in tasks
        for j in range(t.X.shape[1])
        if np.nanstd(t.X[:, j]) > 0
    )
    assert max_scale / min_scale > 1e4, "scale diversity too narrow"


def test_reproducible():
    a = TaskSampler(seed=123).sample()
    b = TaskSampler(seed=123).sample()
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)
    assert a.family == b.family and a.train_size == b.train_size


def test_batch_collates():
    from tabum.train import collate_tasks

    # rejection-prone config (small rows, many classes): retries within a
    # batch must preserve the shared shape, or collation breaks mid-training
    cfg = GeneratorConfig(min_rows=60, max_rows=256, max_features=20, max_classes=10)
    sampler = TaskSampler(cfg, seed=5)
    for _ in range(15):
        batch = collate_tasks(sampler.sample_batch(4))
        assert batch.x.shape[0] == 4
        assert batch.y_train.shape == (4, batch.train_size)
        assert batch.x.shape[1] == batch.train_size + batch.y_test.shape[1]
