"""Scikit-learn-compatible wrappers.

Note on semantics: fit() performs NO weight updates. In the in-context-
learning paradigm, "fitting" just stores (and label-encodes) the training data;
the pretrained transformer conditions on it in a single forward pass at
predict time. Inputs are numeric numpy arrays; NaN marks missing values and is
handled natively by the model (no imputation needed). Categorical columns
should be integer-encoded upstream.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from ..model import ModelConfig, TabUM


class _BaseEstimator:
    def __init__(self, model: TabUM | None = None, checkpoint: str | None = None,
                 device: str | None = None, n_ensemble: int = 1):
        # n_ensemble > 1 averages predictions over permuted views of the data
        # (column order + class indices). The signal is identical in every
        # view; the permutation-sensitivity noise (feature-triplet grouping,
        # label-embedding assignment) is not, so averaging cancels it.
        self.n_ensemble = n_ensemble
        if model is None and checkpoint is None:
            model = TabUM(ModelConfig())  # untrained — useful for tests only
        if model is None:
            from ..train.loop import Trainer

            model = Trainer.load_model(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] != len(y):
            raise ValueError(f"bad shapes: X {X.shape}, y {np.shape(y)}")
        self._X, self._y = X, np.asarray(y)
        return self

    def finetune(self, X: np.ndarray, y: np.ndarray, *, steps: int = 300,
                 lr: float = 3e-5, val_frac: float = 0.15, patience: int = 6,
                 eval_every: int = 10, seed: int = 0, verbose: bool = False):
        """fit() plus a short gradient adaptation of a *cloned* model to this
        dataset: repeatedly pseudo-split the training rows into context and
        targets and train on the same ICL objective as pretraining. A held-out
        slice (never used as targets) drives early stopping; the best state
        wins. The base checkpoint is untouched.
        """
        self.fit(X, y)
        task_type, y_num, n_classes = self._icl_target()
        model = copy.deepcopy(self.model).train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        rng = np.random.default_rng(seed)

        n = len(y_num)
        perm = rng.permutation(n)
        n_val = int(np.clip(round(val_frac * n), 5, 500))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        def val_loss() -> float:
            model.eval()
            with torch.inference_mode():
                loss = self._icl_loss(model, y_num, tr_idx, val_idx,
                                      task_type, n_classes)
            model.train()
            return float("inf") if loss is None else float(loss)

        best = val_loss()
        best_state = copy.deepcopy(model.state_dict())
        bad = 0
        amp = self.device == "cuda"
        for step in range(1, steps + 1):
            order = rng.permutation(tr_idx)
            cut = max(1, int(0.8 * len(order)))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = self._icl_loss(model, y_num, order[:cut], order[cut:],
                                      task_type, n_classes)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % eval_every == 0:
                v = val_loss()
                if verbose:
                    print(f"finetune step {step}: train {float(loss):.4f} val {v:.4f}")
                if v < best:
                    best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        model.load_state_dict(best_state)
        self.model = model.eval()
        return self

    def feature_importances(self, X_test: np.ndarray) -> np.ndarray:
        """Column-ablation attribution: importance of feature j = mean effect on
        the prediction of removing column j from BOTH the fitted context and the
        test rows — a leave-one-covariate-out counterfactual. Retraining-free
        because fit() stores data only; costs one predict per feature.
        Caveat (shared by all perturbation methods): correlated features split
        credit — a feature can score low if another column carries the same signal.
        """
        if self._X is None:
            raise RuntimeError("call fit() first")
        X_test = np.asarray(X_test, dtype=np.float32)
        ref = self._point_scores(X_test)
        saved = self._X
        imp = np.empty(saved.shape[1], dtype=np.float64)
        try:
            for j in range(saved.shape[1]):
                self._X = np.delete(saved, j, axis=1)
                imp[j] = self._ablation_delta(ref, np.delete(X_test, j, axis=1))
        finally:
            self._X = saved
        return imp

    def _icl_loss(self, model: TabUM, y_num: np.ndarray, ctx_idx: np.ndarray,
                  tgt_idx: np.ndarray, task_type: str,
                  n_classes: int | None) -> torch.Tensor | None:
        if len(tgt_idx) == 0:
            return None
        x = torch.from_numpy(
            np.concatenate([self._X[ctx_idx], self._X[tgt_idx]], axis=0)
        ).unsqueeze(0).to(self.device)
        if task_type == "classification":
            seen = np.unique(y_num[ctx_idx])
            mask = np.isin(y_num[tgt_idx], seen)
            if seen.size < 2 or not mask.any():
                return None
            y_ctx = torch.from_numpy(y_num[ctx_idx]).unsqueeze(0).to(self.device)
            out = model(x, y_ctx, len(ctx_idx), "classification", n_classes=n_classes)
            probs = out["probs"][0].float().clamp(min=1e-9)
            tgt = torch.from_numpy(y_num[tgt_idx]).to(self.device)
            m = torch.from_numpy(mask).to(self.device)
            return F.nll_loss(probs.log()[m], tgt[m])
        y_ctx = torch.from_numpy(y_num[ctx_idx]).unsqueeze(0).to(self.device)
        out = model(x, y_ctx, len(ctx_idx), "regression")
        y_tgt = torch.from_numpy(y_num[tgt_idx]).unsqueeze(0).to(self.device)
        y_std = (y_tgt - out["y_mean"]) / out["y_std"]
        return model.reg_head.nll(out["logits"].float(), y_std)

    test_chunk = 8192  # max test rows per forward pass — bounds peak memory

    def _forward(self, X_test: np.ndarray, task_type: str, y_train: torch.Tensor,
                 n_classes: int | None = None) -> dict[str, torch.Tensor]:
        if self._X is None:
            raise RuntimeError("call fit() first")
        X_test = np.asarray(X_test, dtype=np.float32)
        if self.n_ensemble <= 1:
            return self._forward_view(X_test, task_type, y_train, n_classes,
                                      None, None)
        rng = np.random.default_rng(0)
        views = []
        for e in range(self.n_ensemble):
            feat_perm = rng.permutation(self._X.shape[1]) if e else None
            class_perm = (rng.permutation(n_classes)
                          if e and task_type == "classification" else None)
            views.append(self._forward_view(X_test, task_type, y_train,
                                            n_classes, feat_perm, class_perm))
        if task_type == "classification":
            return {"probs": torch.stack([v["probs"] for v in views]).mean(0)}
        # average bar-distribution probabilities; log(mean prob) re-enters the
        # heads as logits exactly (softmax of log-probabilities is identity)
        probs = torch.stack(
            [torch.softmax(v["logits"].float(), -1) for v in views]).mean(0)
        return {**views[0], "logits": probs.clamp_min(1e-12).log()}

    def _forward_view(self, X_test: np.ndarray, task_type: str,
                      y_train: torch.Tensor, n_classes: int | None,
                      feat_perm: np.ndarray | None,
                      class_perm: np.ndarray | None) -> dict[str, torch.Tensor]:
        X_train = self._X
        if feat_perm is not None:
            X_train, X_test = X_train[:, feat_perm], X_test[:, feat_perm]
        if class_perm is not None:
            y_train = torch.from_numpy(
                class_perm.astype(np.int64))[y_train]
        outs = []
        for i in range(0, len(X_test), self.test_chunk):
            x = torch.from_numpy(
                np.concatenate([X_train, X_test[i : i + self.test_chunk]], axis=0)
            ).unsqueeze(0)
            with torch.inference_mode():
                outs.append(self.model(
                    x.to(self.device), y_train.unsqueeze(0).to(self.device),
                    train_size=X_train.shape[0], task_type=task_type,
                    n_classes=n_classes,
                ))
        if len(outs) == 1:
            out = dict(outs[0])
        else:
            key = "probs" if task_type == "classification" else "logits"
            out = {key: torch.cat([o[key] for o in outs], dim=1)}
            for extra in ("y_mean", "y_std"):  # identical across chunks (train-only stats)
                if extra in outs[0]:
                    out[extra] = outs[0][extra]
        if class_perm is not None:
            # view class class_perm[c] is original class c — gather back
            out["probs"] = out["probs"][..., torch.from_numpy(class_perm)]
        return out


class TabUMClassifier(_BaseEstimator):
    def fit(self, X: np.ndarray, y: np.ndarray):
        super().fit(X, y)
        self.classes_, self._y_enc = np.unique(self._y, return_inverse=True)
        if self.classes_.size > self.model.cfg.max_classes:
            raise ValueError(f"{self.classes_.size} classes exceeds model max "
                             f"{self.model.cfg.max_classes}")
        return self

    def _icl_target(self) -> tuple[str, np.ndarray, int | None]:
        return "classification", self._y_enc.astype(np.int64), self.classes_.size

    def _point_scores(self, X_test: np.ndarray) -> np.ndarray:
        return self.predict_proba(X_test)

    def _ablation_delta(self, ref: np.ndarray, X_test_ablated: np.ndarray) -> float:
        pred = ref.argmax(1)
        probs = self.predict_proba(X_test_ablated)
        rows = np.arange(len(pred))
        return float(np.mean(ref[rows, pred] - probs[rows, pred]))

    def explain(self, X_test: np.ndarray, top_k: int = 5) -> dict:
        """Explain predictions two ways, both faithful to the actual computation:
        - feature_importances (F,): drop in predicted-class probability when a
          column is removed from the world (context + test rows)
        - neighbors: per test row, the top_k training rows the retrieval head
          attended to, with their softmax weights and labels — these ARE the
          votes the prediction was computed from
        """
        imp = self.feature_importances(X_test)
        X_test = np.asarray(X_test, dtype=np.float32)
        n_train = self._X.shape[0]
        ws, idxs = [], []
        for i in range(0, len(X_test), self.test_chunk):
            x = torch.from_numpy(
                np.concatenate([self._X, X_test[i : i + self.test_chunk]], axis=0)
            ).unsqueeze(0).to(self.device)
            train_mask = torch.zeros(1, x.shape[1], dtype=torch.bool, device=self.device)
            train_mask[:, :n_train] = True
            y_full = torch.zeros(1, x.shape[1], dtype=torch.int64, device=self.device)
            y_full[:, :n_train] = torch.from_numpy(self._y_enc.astype(np.int64))
            with torch.inference_mode():
                emb = self.model.embed_rows(x, y_full, train_mask, n_train,
                                            "classification")
                w, idx = self.model.cls_head.top_neighbors(
                    emb[:, n_train:], emb[:, :n_train], top_k=top_k)
            ws.append(w[0].float().cpu().numpy())
            idxs.append(idx[0].cpu().numpy())
        idx = np.concatenate(idxs)
        return {
            "feature_importances": imp,
            "neighbor_index": idx,                      # (n_test, top_k) into fit rows
            "neighbor_weight": np.concatenate(ws),      # attention mass per neighbor
            "neighbor_label": self.classes_[self._y_enc[idx]],
        }

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        out = self._forward(X_test, "classification",
                            torch.from_numpy(self._y_enc.astype(np.int64)),
                            n_classes=self.classes_.size)
        return out["probs"][0].float().cpu().numpy()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X_test).argmax(axis=1)]


class TabUMRegressor(_BaseEstimator):
    def _icl_target(self) -> tuple[str, np.ndarray, int | None]:
        return "regression", np.asarray(self._y, dtype=np.float32), None

    def _point_scores(self, X_test: np.ndarray) -> np.ndarray:
        return self.predict(X_test)

    def _ablation_delta(self, ref: np.ndarray, X_test_ablated: np.ndarray) -> float:
        scale = float(np.asarray(self._y, dtype=np.float64).std()) or 1.0
        return float(np.mean(np.abs(ref - self.predict(X_test_ablated)))) / scale

    def explain(self, X_test: np.ndarray, top_k: int = 5) -> dict:
        """Column-ablation feature importances (mean |prediction shift| when a
        column is removed, in train-target-std units). The regression head has
        no retrieval attention, so no neighbor explanations here."""
        return {"feature_importances": self.feature_importances(X_test)}

    def _out(self, X_test: np.ndarray) -> dict[str, torch.Tensor]:
        return self._forward(X_test, "regression",
                             torch.from_numpy(np.asarray(self._y, dtype=np.float32)))

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        out = self._out(X_test)
        return self.model.predict_mean(out)[0].float().cpu().numpy()

    def predict_quantile(self, X_test: np.ndarray, q: float) -> np.ndarray:
        out = self._out(X_test)
        return self.model.predict_quantile(out, q)[0].float().cpu().numpy()
