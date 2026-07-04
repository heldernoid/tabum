"""End-to-end shape/wiring checks: generator -> model -> loss -> estimator."""

import numpy as np
import torch

from tabum.generator import GeneratorConfig, TaskSampler
from tabum.inference import TabUMClassifier, TabUMRegressor
from tabum.model import ModelConfig, TabUM
from tabum.train import TrainConfig, Trainer, collate_tasks
from tabum.train.loop import compute_loss


def _sampler():
    return TaskSampler(GeneratorConfig(max_rows=200, max_features=15), seed=9)


def test_forward_and_loss_both_task_types():
    model = TabUM(ModelConfig.toy())
    sampler = _sampler()
    seen = set()
    while seen != {"classification", "regression"}:
        batch = collate_tasks(sampler.sample_batch(2))
        seen.add(batch.task_type)
        loss = compute_loss(model, batch)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()  # gradients flow end to end
        model.zero_grad(set_to_none=True)


def test_classification_probs_valid():
    model = TabUM(ModelConfig.toy()).eval()
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.standard_normal((2, 30, 5)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, 4, size=(2, 20)))
    with torch.inference_mode():
        out = model(x, y, 20, "classification", n_classes=4)
    probs = out["probs"]
    assert probs.shape == (2, 10, 4)
    torch.testing.assert_close(probs.sum(-1), torch.ones(2, 10), atol=1e-5, rtol=0)
    assert (probs >= 0).all()


def test_regression_outputs():
    model = TabUM(ModelConfig.toy()).eval()
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.standard_normal((1, 40, 6)), dtype=torch.float32)
    y = torch.tensor(rng.standard_normal((1, 30)), dtype=torch.float32)
    with torch.inference_mode():
        out = model(x, y, 30, "regression")
        mean = model.predict_mean(out)
        q10 = model.predict_quantile(out, 0.1)
        q90 = model.predict_quantile(out, 0.9)
    assert mean.shape == (1, 10)
    assert torch.isfinite(mean).all()
    assert (q10 <= q90 + 1e-6).all()


def test_estimator_roundtrip():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((50, 4)).astype(np.float32)
    X[0, 1] = np.nan
    y_cls = rng.integers(0, 3, size=50)
    clf = TabUMClassifier(model=TabUM(ModelConfig.toy()), device="cpu")
    clf.fit(X, y_cls)
    proba = clf.predict_proba(rng.standard_normal((7, 4)))
    assert proba.shape == (7, 3)
    assert set(clf.predict(rng.standard_normal((7, 4)))) <= set(np.unique(y_cls))

    reg = TabUMRegressor(model=TabUM(ModelConfig.toy()), device="cpu")
    reg.fit(X, rng.standard_normal(50))
    pred = reg.predict(rng.standard_normal((7, 4)))
    assert pred.shape == (7,) and np.isfinite(pred).all()


def test_checkpoint_roundtrip(tmp_path):
    model = TabUM(ModelConfig.toy())
    trainer = Trainer(model, TrainConfig(device="cpu", checkpoint_dir=str(tmp_path)))
    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path)
    restored = Trainer.load_model(path)
    for (n1, p1), (n2, p2) in zip(
        model.state_dict().items(), restored.state_dict().items()
    ):
        assert n1 == n2
        torch.testing.assert_close(p1, p2, atol=0, rtol=0)


def test_default_param_budget():
    n = TabUM(ModelConfig()).n_parameters()
    assert 10e6 < n < 20e6, f"default config has {n/1e6:.1f}M params, outside 10-20M target"


def test_retrieval_head_chunk_equivalence():
    """Chunked retrieval must be bit-compatible with the full matrix."""
    model = TabUM(ModelConfig.toy()).eval()
    rng = np.random.default_rng(1)
    te = torch.tensor(rng.standard_normal((1, 50, model.cfg.d_model)), dtype=torch.float32)
    tr = torch.tensor(rng.standard_normal((1, 80, model.cfg.d_model)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, 5, size=(1, 80)))
    with torch.inference_mode():
        full = model.cls_head(te, tr, y, 5, chunk=10_000)
        chunked = model.cls_head(te, tr, y, 5, chunk=7)
    torch.testing.assert_close(full, chunked, atol=1e-6, rtol=0)


def test_estimator_test_chunking():
    """Predictions must not depend on the test_chunk size."""
    rng = np.random.default_rng(4)
    X, y = rng.standard_normal((60, 5)).astype(np.float32), rng.integers(0, 3, size=60)
    Xte = rng.standard_normal((25, 5)).astype(np.float32)
    clf = TabUMClassifier(model=TabUM(ModelConfig.toy()), device="cpu").fit(X, y)
    p_full = clf.predict_proba(Xte)
    clf.test_chunk = 7
    p_chunked = clf.predict_proba(Xte)
    np.testing.assert_allclose(p_full, p_chunked, atol=1e-6)

    reg = TabUMRegressor(model=TabUM(ModelConfig.toy()), device="cpu").fit(
        X, rng.standard_normal(60))
    r_full = reg.predict(Xte)
    reg.test_chunk = 9
    np.testing.assert_allclose(r_full, reg.predict(Xte), atol=1e-5)
