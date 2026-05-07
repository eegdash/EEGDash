# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Baseline recipe classes for Workstream 5.

These wrappers provide a uniform ``.fit(X, y).score(X, y)`` interface plus a
``.predict`` method and a ``requires`` class attribute documenting optional
dependencies. They are intentionally minimal: each recipe thunks to either
scikit-learn or Braindecode, then returns a standardized score dict containing
the headline metric, the chance level, and the metric name.

The ``score`` dict is shaped for tutorial reporting:

``{"score": float, "chance_level": float, "metric": str}``

For regression baselines, an additional ``"mse"`` and ``"r2"`` are included.

Optional dependencies are imported lazily inside class methods (never at module
load) so the rest of ``eegdash`` keeps a small footprint and graceful failure
when, for example, ``lightgbm`` or ``torch`` are not installed.

References
----------
- Schirrmeister et al. (2017). Deep learning with convolutional neural networks
  for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391-5420.
  doi:10.1002/hbm.23730
- Lawhern et al. (2018). EEGNet: a compact convolutional neural network for
  EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5),
  056013. doi:10.1088/1741-2552/aace8c

"""

from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np

from ._baselines import majority_baseline, median_baseline

ArrayLike = Union[Sequence, np.ndarray]


def _to_numpy_2d(values: ArrayLike) -> np.ndarray:
    """Coerce an array-like into a 2D numpy float array (n_samples, n_features)."""
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr.astype(np.float64, copy=False)


def _to_numpy_1d(values: ArrayLike) -> np.ndarray:
    """Coerce an array-like into a 1D numpy array."""
    arr = np.asarray(values)
    return arr.reshape(-1)


def _missing_dep_error(name: str, package: str) -> ImportError:
    return ImportError(
        f"{name} requires the optional dependency '{package}'. "
        f"Install it with: pip install {package}"
    )


class _BaseRecipe:
    """Common behaviour for the baseline recipes."""

    requires: tuple[str, ...] = ()
    metric: str = "score"

    def __init__(self, random_state: int = 42, **kwargs: Any) -> None:
        self.random_state = random_state
        self.kwargs = dict(kwargs)
        self._model: Any = None
        self._y_train: np.ndarray | None = None

    # API surface; subclasses override the actual training and scoring.
    def fit(self, X: ArrayLike, y: ArrayLike) -> "_BaseRecipe":  # pragma: no cover
        raise NotImplementedError

    def predict(self, X: ArrayLike) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def score(self, X: ArrayLike, y: ArrayLike) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Classical baselines (scikit-learn)
# ---------------------------------------------------------------------------


class RidgeRegressionBaseline(_BaseRecipe):
    """Ridge regression baseline backed by ``sklearn.linear_model.Ridge``.

    Suitable for tutorials that need a quick, interpretable regression baseline
    over feature matrices (n_samples, n_features). The chance level is computed
    via :func:`eegdash.splits.median_baseline` so the reported R^2 can be
    compared against a constant predictor.
    """

    requires: tuple[str, ...] = ("scikit-learn",)
    metric: str = "r2"

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RidgeRegressionBaseline":
        from sklearn.linear_model import Ridge

        X_arr = _to_numpy_2d(X)
        y_arr = _to_numpy_1d(y).astype(np.float64, copy=False)
        alpha = float(self.kwargs.get("alpha", 1.0))
        self._model = Ridge(alpha=alpha, random_state=self.random_state)
        self._model.fit(X_arr, y_arr)
        self._y_train = y_arr
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("RidgeRegressionBaseline has not been fitted yet.")
        return np.asarray(self._model.predict(_to_numpy_2d(X)))

    def score(self, X: ArrayLike, y: ArrayLike) -> dict[str, Any]:
        from sklearn.metrics import mean_squared_error, r2_score

        if self._model is None or self._y_train is None:
            raise RuntimeError("RidgeRegressionBaseline has not been fitted yet.")
        y_arr = _to_numpy_1d(y).astype(np.float64, copy=False)
        y_pred = self.predict(X)
        r2 = float(r2_score(y_arr, y_pred))
        mse = float(mean_squared_error(y_arr, y_pred))
        chance = median_baseline(self._y_train, y_arr)
        return {
            "score": r2,
            "r2": r2,
            "mse": mse,
            "chance_level": float(chance["chance_level"]),
            "metric": self.metric,
        }


class LogisticRegressionBaseline(_BaseRecipe):
    """Logistic regression baseline backed by ``sklearn.linear_model``.

    The chance level is the proportion of the majority class on the test set,
    computed by :func:`eegdash.splits.majority_baseline`.
    """

    requires: tuple[str, ...] = ("scikit-learn",)
    metric: str = "accuracy"

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegressionBaseline":
        from sklearn.linear_model import LogisticRegression

        X_arr = _to_numpy_2d(X)
        y_arr = _to_numpy_1d(y)
        max_iter = int(self.kwargs.get("max_iter", 200))
        self._model = LogisticRegression(
            max_iter=max_iter,
            random_state=self.random_state,
            **{k: v for k, v in self.kwargs.items() if k != "max_iter"},
        )
        self._model.fit(X_arr, y_arr)
        self._y_train = y_arr
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LogisticRegressionBaseline has not been fitted yet.")
        return np.asarray(self._model.predict(_to_numpy_2d(X)))

    def score(self, X: ArrayLike, y: ArrayLike) -> dict[str, Any]:
        from sklearn.metrics import accuracy_score

        if self._model is None or self._y_train is None:
            raise RuntimeError("LogisticRegressionBaseline has not been fitted yet.")
        y_arr = _to_numpy_1d(y)
        y_pred = self.predict(X)
        accuracy = float(accuracy_score(y_arr, y_pred))
        chance = majority_baseline(self._y_train, y_arr)
        return {
            "score": accuracy,
            "accuracy": accuracy,
            "chance_level": float(chance["chance_level"]),
            "metric": self.metric,
        }


class LightGBMBaseline(_BaseRecipe):
    """Gradient-boosted decision tree baseline backed by LightGBM.

    LightGBM is an optional dependency. If it is not installed, calling
    :meth:`fit` raises ``ImportError`` with a friendly message rather than
    failing at module import time. CPU defaults are tuned for tutorial-friendly
    runtime: 100 estimators, learning rate 0.1.
    """

    requires: tuple[str, ...] = ("lightgbm",)
    metric: str = "accuracy"

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LightGBMBaseline":
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - exercised in tests
            raise _missing_dep_error("LightGBMBaseline", "lightgbm") from exc

        X_arr = _to_numpy_2d(X)
        y_arr = _to_numpy_1d(y)
        params = {
            "n_estimators": int(self.kwargs.get("n_estimators", 100)),
            "learning_rate": float(self.kwargs.get("learning_rate", 0.1)),
            "num_leaves": int(self.kwargs.get("num_leaves", 31)),
            "n_jobs": int(self.kwargs.get("n_jobs", 1)),
            "verbose": int(self.kwargs.get("verbose", -1)),
            "random_state": self.random_state,
        }
        self._model = lgb.LGBMClassifier(**params)
        self._model.fit(X_arr, y_arr)
        self._y_train = y_arr
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LightGBMBaseline has not been fitted yet.")
        return np.asarray(self._model.predict(_to_numpy_2d(X)))

    def score(self, X: ArrayLike, y: ArrayLike) -> dict[str, Any]:
        from sklearn.metrics import accuracy_score

        if self._model is None or self._y_train is None:
            raise RuntimeError("LightGBMBaseline has not been fitted yet.")
        y_arr = _to_numpy_1d(y)
        y_pred = self.predict(X)
        accuracy = float(accuracy_score(y_arr, y_pred))
        chance = majority_baseline(self._y_train, y_arr)
        return {
            "score": accuracy,
            "accuracy": accuracy,
            "chance_level": float(chance["chance_level"]),
            "metric": self.metric,
        }


# ---------------------------------------------------------------------------
# Neural baselines (Braindecode)
# ---------------------------------------------------------------------------


def _coerce_eeg_tensor(X: ArrayLike) -> Any:
    """Convert (n_samples, n_channels, n_times) input into a torch tensor."""
    import torch  # local import; avoid module-level torch dependency

    if isinstance(X, torch.Tensor):
        tensor = X
    else:
        tensor = torch.as_tensor(np.asarray(X))
    if tensor.ndim != 3:
        raise ValueError(
            "Braindecode baselines expect 3D input "
            f"(n_samples, n_channels, n_times); got shape {tuple(tensor.shape)}."
        )
    return tensor.float()


def _coerce_targets(y: ArrayLike) -> Any:
    import torch

    arr = np.asarray(y).reshape(-1)
    return torch.as_tensor(arr, dtype=torch.long)


class _BraindecodeBaseline(_BaseRecipe):
    """Shared training scaffold for Braindecode-based baselines."""

    requires: tuple[str, ...] = ("braindecode", "torch")
    metric: str = "accuracy"
    _model_cls_name: str = ""

    def _build_model(self, n_chans: int, n_outputs: int, n_times: int) -> Any:
        raise NotImplementedError

    def fit(self, X: ArrayLike, y: ArrayLike) -> "_BraindecodeBaseline":
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:  # pragma: no cover - exercised in tests
            raise _missing_dep_error(self.__class__.__name__, "torch") from exc
        try:
            import braindecode  # noqa: F401  (presence check)
        except ImportError as exc:  # pragma: no cover - exercised in tests
            raise _missing_dep_error(self.__class__.__name__, "braindecode") from exc

        torch.manual_seed(int(self.random_state))

        X_t = _coerce_eeg_tensor(X)
        y_t = _coerce_targets(y)
        n_samples, n_chans, n_times = X_t.shape
        classes = sorted(np.unique(y_t.cpu().numpy()).tolist())
        n_outputs = max(2, len(classes))

        model = self._build_model(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
        device = torch.device("cpu")
        model = model.to(device)

        epochs = int(self.kwargs.get("epochs", 1))
        batch_size = int(self.kwargs.get("batch_size", min(16, n_samples)))
        lr = float(self.kwargs.get("lr", 1e-3))
        weight_decay = float(self.kwargs.get("weight_decay", 1e-4))

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(
            dataset,
            batch_size=max(1, batch_size),
            shuffle=True,
            generator=torch.Generator().manual_seed(int(self.random_state)),
        )

        model.train()
        for _ in range(max(1, epochs)):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb.to(device))
                if logits.ndim == 3:
                    # Braindecode models may emit (B, C, 1) for crop=False.
                    logits = logits.mean(dim=-1)
                loss = loss_fn(logits, yb.to(device))
                loss.backward()
                optimizer.step()

        self._model = model
        self._y_train = np.asarray(y).reshape(-1)
        self._classes = np.asarray(classes)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError(f"{self.__class__.__name__} has not been fitted yet.")
        X_t = _coerce_eeg_tensor(X)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t)
            if logits.ndim == 3:
                logits = logits.mean(dim=-1)
            preds = logits.argmax(dim=1).cpu().numpy()
        # Map argmax indices back to class labels.
        return np.asarray(self._classes)[preds]

    def score(self, X: ArrayLike, y: ArrayLike) -> dict[str, Any]:
        from sklearn.metrics import accuracy_score

        if self._model is None or self._y_train is None:
            raise RuntimeError(f"{self.__class__.__name__} has not been fitted yet.")
        y_arr = _to_numpy_1d(y)
        y_pred = self.predict(X)
        accuracy = float(accuracy_score(y_arr, y_pred))
        chance = majority_baseline(self._y_train, y_arr)
        return {
            "score": accuracy,
            "accuracy": accuracy,
            "chance_level": float(chance["chance_level"]),
            "metric": self.metric,
        }


class ShallowFBCSPNetBaseline(_BraindecodeBaseline):
    """ShallowFBCSPNet baseline (Schirrmeister et al., 2017).

    Wraps :class:`braindecode.models.ShallowFBCSPNet` with a tiny AdamW training
    loop. Defaults run for one epoch with batch size 16 so the recipe is fast
    enough for tutorials but still exercises the full Braindecode codepath.

    References
    ----------
    Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter,
    M., Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W., & Ball, T.
    (2017). Deep learning with convolutional neural networks for EEG decoding
    and visualization. *Human Brain Mapping*, 38(11), 5391-5420.
    doi:10.1002/hbm.23730

    """

    _model_cls_name = "ShallowFBCSPNet"

    def _build_model(self, n_chans: int, n_outputs: int, n_times: int) -> Any:
        from braindecode.models import ShallowFBCSPNet

        # Allow user overrides for any constructor kwarg via self.kwargs.
        kwargs = {
            k: v
            for k, v in self.kwargs.items()
            if k not in {"epochs", "batch_size", "lr", "weight_decay"}
        }
        return ShallowFBCSPNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            final_conv_length="auto",
            **kwargs,
        )


class EEGNetv4Baseline(_BraindecodeBaseline):
    """EEGNetv4 baseline (Lawhern et al., 2018).

    Wraps :class:`braindecode.models.EEGNetv4` with the same minimal AdamW
    training loop used by :class:`ShallowFBCSPNetBaseline`.

    References
    ----------
    Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P.,
    & Lance, B. J. (2018). EEGNet: a compact convolutional neural network for
    EEG-based brain-computer interfaces. *Journal of Neural Engineering*,
    15(5), 056013. doi:10.1088/1741-2552/aace8c

    """

    _model_cls_name = "EEGNetv4"

    def _build_model(self, n_chans: int, n_outputs: int, n_times: int) -> Any:
        from braindecode.models import EEGNetv4

        kwargs = {
            k: v
            for k, v in self.kwargs.items()
            if k not in {"epochs", "batch_size", "lr", "weight_decay"}
        }
        return EEGNetv4(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            final_conv_length="auto",
            **kwargs,
        )


__all__ = [
    "EEGNetv4Baseline",
    "LightGBMBaseline",
    "LogisticRegressionBaseline",
    "RidgeRegressionBaseline",
    "ShallowFBCSPNetBaseline",
]
