"""The four invariance/leakage tests from PLAN.md Phase 2.

These are the most important tests in the repo (see AGENTS.md): a failure here
means the model silently learns spurious structure or leaks information, which
looks fine on i.i.d. eval and fails in deployment. Run in float64 so "equal up
to float tolerance" is meaningful and not masked by fp32 attention noise.
"""

import numpy as np
import pytest
import torch

from tabum.model import ModelConfig, TabUM
from tabum.model.preprocessing import build_groups

TRAIN, TEST, F = 40, 12, 7
ATOL = 1e-8


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    model = TabUM(ModelConfig.toy()).double().eval()
    rng = np.random.default_rng(42)
    x = torch.tensor(rng.standard_normal((1, TRAIN + TEST, F)))
    x[0, 3, 2] = torch.nan  # include missing values in the invariance checks
    x[0, TRAIN + 2, 5] = torch.nan
    y_cls = torch.tensor(rng.integers(0, 3, size=(1, TRAIN)))
    y_reg = torch.tensor(rng.standard_normal((1, TRAIN)))
    return model, x, y_cls, y_reg


def _predict(model, x, y_cls, groups=None):
    with torch.inference_mode():
        return model(x, y_cls, TRAIN, "classification", n_classes=3, groups=groups)["probs"]


def test_row_permutation_invariance(setup):
    """Shuffling train-row order must not change test predictions."""
    model, x, y_cls, _ = setup
    base = _predict(model, x, y_cls)
    perm = torch.randperm(TRAIN)
    x_p = torch.cat([x[:, perm], x[:, TRAIN:]], dim=1)
    shuffled = _predict(model, x_p, y_cls[:, perm])
    torch.testing.assert_close(base, shuffled, atol=ATOL, rtol=0)


def test_column_permutation_invariance(setup):
    """Shuffling column order (with the grouping remapped so each triplet
    contains the same features) must not change predictions."""
    model, x, y_cls, _ = setup
    groups = build_groups(F, seed=7)
    base = _predict(model, x, y_cls, groups=groups)
    perm = torch.randperm(F)
    inv = torch.argsort(perm)
    groups_p = torch.where(groups >= 0, inv[groups.clamp(min=0)], groups)
    shuffled = _predict(model, x[:, :, perm], y_cls, groups=groups_p)
    torch.testing.assert_close(base, shuffled, atol=ATOL, rtol=0)


def test_no_test_row_leakage(setup):
    """Perturbing test row B must not change predictions for test row A."""
    model, x, y_cls, _ = setup
    base = _predict(model, x, y_cls)
    x_p = x.clone()
    x_p[0, TRAIN + 1] += 100.0  # blow up one test row
    x_p[0, TRAIN + 5] = torch.nan  # and NaN-bomb another
    pert = _predict(model, x_p, y_cls)
    others = [i for i in range(TEST) if i not in (1, 5)]
    torch.testing.assert_close(base[:, others], pert[:, others], atol=ATOL, rtol=0)
    # sanity: the perturbed rows themselves SHOULD change
    assert not torch.allclose(base[:, 1], pert[:, 1], atol=1e-4)


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_no_label_leakage(setup, task_type):
    """Label embeddings must only touch train rows: garbage labels at test
    positions of y_full must leave the row embeddings bit-identical."""
    model, x, y_cls, y_reg = setup
    y_train = y_cls if task_type == "classification" else y_reg
    train_mask = torch.zeros(1, TRAIN + TEST, dtype=torch.bool)
    train_mask[:, :TRAIN] = True
    y_full = torch.zeros(1, TRAIN + TEST, dtype=y_train.dtype)
    y_full[:, :TRAIN] = y_train
    y_garbage = y_full.clone()
    if task_type == "classification":
        y_garbage[:, TRAIN:] = 2  # a real class id, so leakage would be plausible
    else:
        y_garbage[:, TRAIN:] = 1e6
    with torch.inference_mode():
        a = model.embed_rows(x, y_full, train_mask, TRAIN, task_type)
        b = model.embed_rows(x, y_garbage, train_mask, TRAIN, task_type)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    # sanity: changing TRAIN labels must change embeddings (labels do flow)
    y_alt = y_full.clone()
    y_alt[:, 0] = 1 if task_type == "classification" else 5.0
    with torch.inference_mode():
        c = model.embed_rows(x, y_alt, train_mask, TRAIN, task_type)
    assert not torch.allclose(a, c, atol=1e-6)
