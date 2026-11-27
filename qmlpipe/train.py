"""Training and evaluation utilities for the rebuilt hybrid QNN pipeline (QML-D02).

The module is deliberately configuration- and data-agnostic: a caller supplies a
model, two DataLoaders and a config object exposing the QMLConfig fields listed in
the swarm contract.  No dataset path, no paper number and no PennyLane import is
present here, so the module stays importable in the central verifier's
Python/PennyLane-free environment.  Quantum models are constructed by QML-D01's
``qmlpipe.models`` builders and passed in by the CLI.

QHEAD-D01 additions (all split-safe):

* class-balanced cross entropy / focal loss with class weights derived from
  TRAIN labels only (:mod:`qmlpipe.losses`);
* checkpoint/early-stop selection on a configurable validation metric
  (``val_loss``, ``balanced_accuracy`` or ``macro_f1``);
* an optional class-balanced sampler built from TRAIN labels only;
* a public :func:`load_checkpoint` helper so evaluation uses the selected
  checkpoint rather than the last-epoch weights.

Temperature/logit-adjustment calibration is fitted on ID_VAL only by
:func:`qmlpipe.heads.fit_calibration`, never on OOD, and applied at evaluation by
the model's classification head.  Every choice is recorded in the run's
checkpoint / summary provenance.
"""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from qmlpipe.losses import (
    as_two_logits,
    class_weights_from_labels,
    focal_loss,
    weighted_cross_entropy,
)

try:  # pragma: no cover - a minimal verifier may not install torch
    import torch
    import torch.nn.functional as F

    _TORCH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

__all__ = [
    "seed_everything",
    "train",
    "evaluate",
    "build_balanced_sampler",
    "load_checkpoint",
]

HISTORY_KEYS = (
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
    "lr",
    "val_balanced_accuracy",
    "val_macro_f1",
)
_SUPPORTED_OPTIMIZERS = ("adamw", "adam", "sgd", "spsa")
_SUPPORTED_EARLY_METRICS = ("val_loss", "balanced_accuracy", "macro_f1")


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - depends on the host environment
        raise RuntimeError(
            "qmlpipe.train requires PyTorch; "
            f"import failed with {type(_TORCH_IMPORT_ERROR).__name__}: {_TORCH_IMPORT_ERROR}"
        )


# ---------------------------------------------------------------------------
# Config / serialisation helpers
# ---------------------------------------------------------------------------
def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from an object, its ``as_dict()`` or a mapping."""
    if isinstance(cfg, Mapping) and name in cfg:
        return cfg[name]
    value = getattr(cfg, name, None)
    if value is not None:
        return value
    as_dict = getattr(cfg, "as_dict", None)
    if callable(as_dict):
        try:
            return as_dict().get(name, default)
        except Exception:
            pass
    return default


def _config_snapshot(cfg: Any) -> dict:
    """Return a JSON-native snapshot of the config (never a live object)."""
    as_dict = getattr(cfg, "as_dict", None)
    if callable(as_dict):
        try:
            return json.loads(json.dumps(as_dict(), sort_keys=True, default=str))
        except Exception:
            pass
    if isinstance(cfg, Mapping):
        return json.loads(json.dumps(dict(cfg), sort_keys=True, default=str))
    return {
        key: value
        for key, value in vars(cfg).items()
        if not str(key).startswith("_")
    }


def _git_commit_hash() -> str:
    """Local ``git rev-parse HEAD`` (no network); ``"unavailable"`` otherwise."""
    root = Path(__file__).resolve().parents[1]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    if proc.returncode != 0:
        return "unavailable"
    sha = proc.stdout.strip()
    return sha if sha else "unavailable"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs and make cuDNN deterministic.

    The function is safe to call in environments without torch.  It deliberately
    does not enable ``torch.use_deterministic_algorithms`` globally, which can
    reject PennyLane/TorchLayer kernels; cuDNN determinism is enough for the
    CPU-first quantum pipeline.
    """
    value = int(seed)
    random.seed(value)
    np.random.seed(value % (2 ** 32))
    if torch is None:
        return
    torch.manual_seed(value)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tensor plumbing
# ---------------------------------------------------------------------------
def _model_device_and_dtype(model: Any):
    """Infer device and floating dtype from the model's first parameter."""
    _require_torch()
    try:
        parameter = next(model.parameters())
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu"), torch.float32
    return parameter.device, parameter.dtype if parameter.is_floating_point() else torch.float32


def _prepare_batch(batch: Any, device: Any, dtype: Any):
    """Split a ``(X, y)`` batch, cast features to the model dtype and labels to long."""
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("DataLoader batches must be (features, labels) pairs")
    X, y = batch[0], batch[1]
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X)
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)
    X = X.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=torch.long).reshape(-1)
    return X, y


def _labels_from_loader(loader: Any) -> np.ndarray:
    """Flatten the label sequence of a loader (fallback when no dataset handle)."""
    values: list[np.ndarray] = []
    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            continue
        y = batch[1]
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        values.append(np.asarray(y).reshape(-1))
    if not values:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(values).astype(np.int64)


def _raw_train_labels(loader: Any) -> np.ndarray:
    """Read the underlying TRAIN labels, bypassing any sampler reordering.

    For a ``TensorDataset`` this reads ``dataset.tensors[1]`` directly, so a
    :class:`build_balanced_sampler` does not distort the class-weight estimate.
    Falls back to iterating the loader when no dataset handle is available.
    """
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        for attr in ("labels", "targets", "y"):
            value = getattr(dataset, attr, None)
            if value is not None:
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                return np.asarray(value).reshape(-1).astype(np.int64)
        tensors = getattr(dataset, "tensors", None)
        if tensors is not None and len(tensors) > 1:
            labels = tensors[1]
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
            return np.asarray(labels).reshape(-1).astype(np.int64)
    return _labels_from_loader(loader)


def build_balanced_sampler(labels: Any, seed: int = 42) -> Any:
    """Class-balanced ``WeightedRandomSampler`` derived from TRAIN labels only.

    Each sample is drawn with probability proportional to
    ``1 / count[its_class]``.  Returns ``None`` when the label set is empty, so
    callers can degrade honestly instead of sampling from an undefined
    distribution.  ``seed`` is forwarded to the sampler's own generator.
    """
    _require_torch()
    y = np.asarray(labels).reshape(-1).astype(np.int64)
    if y.size == 0:
        return None
    classes, counts = np.unique(y, return_counts=True)
    per_sample = np.array([1.0 / float(counts[np.where(classes == int(label))[0][0]]) for label in y])
    generator = torch.Generator().manual_seed(int(seed))
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(per_sample, dtype=torch.float64),
        num_samples=int(y.size),
        replacement=True,
        generator=generator,
    )


def _binary_confusion(y_true: Any, y_pred: Any) -> tuple[int, int, int, int]:
    """Return ``(tn, fp, fn, tp)`` for a binary prediction pair (numpy only)."""
    true = np.asarray(y_true).reshape(-1).astype(np.int64)
    pred = np.asarray(y_pred).reshape(-1).astype(np.int64)
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(true, pred):
        if 0 <= int(t) < 2 and 0 <= int(p) < 2:
            cm[int(t), int(p)] += 1
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return tn, fp, fn, tp


def _balanced_accuracy(y_true: Any, y_pred: Any) -> float:
    """Mean per-class recall ``(recall_0 + recall_1) / 2`` (numpy only)."""
    tn, fp, fn, tp = _binary_confusion(y_true, y_pred)
    r0 = tn / (tn + fp) if (tn + fp) else 0.0
    r1 = tp / (tp + fn) if (tp + fn) else 0.0
    return float(0.5 * (r0 + r1))


def _macro_f1(y_true: Any, y_pred: Any) -> float:
    """Unweighted mean of the two per-class F1 scores (numpy only)."""
    tn, fp, fn, tp = _binary_confusion(y_true, y_pred)
    d0 = 2 * tn + fp + fn
    d1 = 2 * tp + fp + fn
    f0 = 2 * tn / d0 if d0 else 0.0
    f1 = 2 * tp / d1 if d1 else 0.0
    return float(0.5 * (f0 + f1))


def _selection_metric(
    name: str, val_loss: float, val_acc: float, y_true: Any, y_pred: Any
) -> tuple[float, bool]:
    """Return ``(value, higher_is_better)`` for a checkpoint-selection metric."""
    key = str(name).strip().lower()
    if key == "val_loss":
        return float(val_loss), False
    if key == "balanced_accuracy":
        return _balanced_accuracy(y_true, y_pred), True
    if key == "macro_f1":
        return _macro_f1(y_true, y_pred), True
    raise ValueError(
        f"unknown early_metric {name!r}; expected one of {_SUPPORTED_EARLY_METRICS}"
    )


def _make_loss_fn(cfg: Any, weights: Any):
    """Build the training loss callable from ``loss``/``class_weight`` config."""
    name = str(_cfg_get(cfg, "loss", "ce")).strip().lower()
    if name == "ce":
        return lambda logits, y: weighted_cross_entropy(logits, y, weight=weights)
    if name == "focal":
        gamma = float(_cfg_get(cfg, "focal_gamma", 2.0))
        alpha = _cfg_get(cfg, "focal_alpha", None)
        alpha = None if alpha is None else float(alpha)
        return lambda logits, y: focal_loss(
            logits, y, weight=weights, gamma=gamma, alpha=alpha
        )
    raise ValueError(f"unknown loss {name!r}; expected 'ce' or 'focal'")


def _binary_outputs(logits: Any, y: Any, loss_fn: Any = None):
    """Return ``(loss, positive_probability, hard_prediction)`` for binary logits.

    Accepts a two-column logit tensor, a single logit column, a one-dimensional
    logit vector, or a tuple/list whose first element is a logit tensor.
    """
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    two_logits = as_two_logits(logits)
    probability = torch.softmax(two_logits.float(), dim=1)[:, 1]
    prediction = torch.argmax(two_logits, dim=1)
    if loss_fn is not None:
        loss = loss_fn(two_logits, y)
    else:
        # Legacy path. For a single logit z, as_two_logits returns [-z/2, z/2],
        # whose cross entropy equals binary_cross_entropy_with_logits(z) exactly.
        loss = F.cross_entropy(two_logits, y)
    return loss, probability, prediction


def _build_optimizer(model: Any, cfg: Any):
    """Construct AdamW, SGD or SPSA from the config's ``optimizer`` field."""
    _require_torch()
    name = str(_cfg_get(cfg, "optimizer", "adamw")).strip().lower()
    lr = float(_cfg_get(cfg, "learning_rate", 1e-3))
    weight_decay = float(_cfg_get(cfg, "weight_decay", 0.0))
    parameters = model.parameters()
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay), False
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay), False
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay), False
    if name == "spsa":
        spsa = getattr(torch.optim, "SPSA", None)
        if spsa is None:  # pragma: no cover - depends on the installed PyTorch
            raise RuntimeError("torch.optim.SPSA is unavailable in this PyTorch build")
        generator = torch.Generator().manual_seed(int(_cfg_get(cfg, "seed", 42)))
        try:
            optimizer = spsa(
                parameters, lr=lr, nprocs=1, last=True, rng=generator
            )
        except TypeError:  # older SPSA signatures have no rng/last keyword
            optimizer = spsa(parameters, lr=lr)
        return optimizer, True
    raise ValueError(
        f"Unknown optimizer {name!r}; expected one of {', '.join(_SUPPORTED_OPTIMIZERS)}"
    )


def _train_step(
    model: Any, X: Any, y: Any, optimizer: Any, cfg: Any, is_spsa: bool, loss_fn: Any = None
):
    """One optimisation step; returns ``(loss_value, predictions)``."""
    if is_spsa:
        # SPSA estimates the gradient from two loss evaluations and therefore
        # takes a closure that returns the loss without calling backward().
        observed: list[Any] = []

        def closure():
            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss, _, prediction = _binary_outputs(logits, y, loss_fn=loss_fn)
            observed.append(loss.detach())
            return loss

        result = optimizer.step(closure)
        if hasattr(result, "detach"):
            loss_value = float(result.detach().cpu())
        elif result is not None:
            loss_value = float(result)
        else:
            loss_value = float(observed[-1].cpu())
        with torch.no_grad():
            _, _, prediction = _binary_outputs(model(X), y, loss_fn=loss_fn)
        return loss_value, prediction.detach()

    optimizer.zero_grad(set_to_none=True)
    logits = model(X)
    loss, _, prediction = _binary_outputs(logits, y, loss_fn=loss_fn)
    loss.backward()
    clip_norm = float(_cfg_get(cfg, "grad_clip", 1.0))
    if clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    return float(loss.detach().cpu()), prediction.detach()


def _epoch_loss_acc(
    model: Any,
    loader: Any,
    cfg: Any,
    optimizer: Any = None,
    is_spsa: bool = False,
    loss_fn: Any = None,
    collect: bool = False,
):
    """Mean loss and accuracy over one pass of ``loader`` (train or eval).

    When ``collect=True`` the return value is a 5-tuple
    ``(loss, accuracy, n, y_true, y_pred)``; the label lists let the caller
    compute balanced accuracy / macro F1 for checkpoint selection.  The default
    3-tuple keeps older callers working unchanged.
    """
    _require_torch()
    device, dtype = _model_device_and_dtype(model)
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            X, y = _prepare_batch(batch, device, dtype)
            if training:
                loss_value, prediction = _train_step(
                    model, X, y, optimizer, cfg, is_spsa, loss_fn=loss_fn
                )
            else:
                logits = model(X)
                loss, _, prediction = _binary_outputs(logits, y, loss_fn=loss_fn)
                loss_value = float(loss.detach().cpu())
            batch_n = int(y.shape[0])
            total_loss += loss_value * batch_n
            total_correct += int((prediction == y).sum().item())
            total_n += batch_n
            if collect:
                y_true.extend(int(v) for v in y.detach().cpu().numpy().tolist())
                y_pred.extend(int(v) for v in prediction.detach().cpu().numpy().tolist())
    if collect:
        if total_n == 0:
            return 0.0, 0.0, 0, [], []
        return total_loss / total_n, total_correct / total_n, total_n, y_true, y_pred
    if total_n == 0:
        return 0.0, 0.0, 0
    return total_loss / total_n, total_correct / total_n, total_n


def _save_checkpoint(
    model: Any,
    cfg: Any,
    epoch: int,
    val_loss: float | None,
    val_accuracy: float | None,
    path: Path,
    extra: Mapping | None = None,
) -> None:
    """Persist the best-so-far weights plus full config and commit provenance."""
    snapshot = _config_snapshot(cfg)
    payload = {
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "epoch": int(epoch),
        "val_loss": None if val_loss is None else float(val_loss),
        "val_accuracy": None if val_accuracy is None else float(val_accuracy),
        "cfg": snapshot,
        "config": snapshot,
        "commit_hash": _git_commit_hash(),
    }
    if extra:
        payload.update(dict(extra))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(model: Any, path: str | Path) -> dict:
    """Load a checkpoint's ``state_dict`` into ``model`` and return its metadata.

    Used by the CLI so OOD evaluation uses the *selected* checkpoint rather than
    the last-epoch weights.  ``weights_only`` is used when the installed torch
    supports it, with a graceful fallback for older builds.
    """
    _require_torch()
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch without weights_only
        payload = torch.load(str(path), map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    return {
        "epoch": payload.get("epoch"),
        "val_loss": payload.get("val_loss"),
        "val_accuracy": payload.get("val_accuracy"),
        "early_metric": payload.get("early_metric"),
        "best_val_metric": payload.get("best_val_metric"),
    }


# ---------------------------------------------------------------------------
# Public training / evaluation API
# ---------------------------------------------------------------------------
def train(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    cfg: Any,
    output_dir: str | Path | None = None,
    history_path: str | Path | None = None,
    ckpt_path: str | Path | None = None,
) -> dict:
    """Train ``model`` with split-safe early stopping and checkpointing.

    Optimizer, learning rate, weight decay, patience and epochs all come from
    ``cfg``.  A full cosine schedule is stepped once per epoch.  Gradients are
    clipped by ``cfg.grad_clip`` (default 1.0) for gradient-based optimizers;
    SPSA uses its closure API and has no backpropagated gradient to clip.

    QHEAD-D01 behaviour, all derived from TRAIN labels only when applicable:

    * ``loss`` in {``ce``, ``focal``} and ``class_weight`` in {``none``,
      ``balanced``} select the training objective and class weights.
    * ``early_metric`` in {``val_loss``, ``balanced_accuracy``, ``macro_f1``}
      selects the checkpoint/early-stop quantity.  ``val_loss`` preserves the
      legacy behaviour; the balanced metrics are the default in the QML CLI.
    * The selected checkpoint records the recipe in its payload.

    Returns a JSON-serialisable dictionary with the full ``history``, the best
    epoch and validation metrics, the selected metric/value, loss provenance,
    the number of epochs run, and whether early stopping fired.
    """
    _require_torch()
    seed_everything(_cfg_get(cfg, "seed", 42))
    epochs = max(1, int(_cfg_get(cfg, "epochs", 15)))
    patience = max(1, int(_cfg_get(cfg, "patience", 5)))
    base_dir = Path(output_dir) if output_dir is not None else Path(
        str(_cfg_get(cfg, "output_dir", "results"))
    )
    history_file = Path(history_path) if history_path else base_dir / "history.json"
    checkpoint_file = Path(ckpt_path) if ckpt_path else base_dir / "checkpoints" / "best.pt"

    loss_name = str(_cfg_get(cfg, "loss", "ce")).strip().lower()
    class_weight = str(_cfg_get(cfg, "class_weight", "none")).strip().lower()
    early_metric = str(_cfg_get(cfg, "early_metric", "val_loss")).strip().lower()
    min_delta = float(_cfg_get(cfg, "early_min_delta", 0.0))
    _, higher_is_better = _selection_metric(early_metric, 0.0, 0.0, [], [])

    # Class weights come from the underlying TRAIN dataset only, never from
    # validation or OOD data. A balanced sampler's reordering is bypassed.
    weights = None
    class_weights_source = "not_used"
    train_class_counts: list[int] | None = None
    if class_weight == "balanced":
        train_labels = _raw_train_labels(train_loader)
        weights = class_weights_from_labels(train_labels, n_classes=2, mode="balanced")
        class_weights_source = "train_labels"
        train_class_counts = [int(v) for v in np.bincount(train_labels, minlength=2)]
    elif class_weight != "none":
        raise ValueError(
            f"unknown class_weight {class_weight!r}; expected 'none' or 'balanced'"
        )
    loss_fn = _make_loss_fn(cfg, weights)

    optimizer, is_spsa = _build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: dict[str, list[float]] = {key: [] for key in HISTORY_KEYS}
    best_score = float("-inf") if higher_is_better else float("inf")
    best_epoch = -1
    best_val_loss: float | None = None
    best_val_acc: float | None = None
    wait = 0
    early_stopped = False

    for epoch in range(epochs):
        train_loss, train_acc, _ = _epoch_loss_acc(
            model, train_loader, cfg, optimizer=optimizer, is_spsa=is_spsa, loss_fn=loss_fn
        )
        if val_loader is not None:
            val_loss, val_acc, val_n, y_true, y_pred = _epoch_loss_acc(
                model, val_loader, cfg, loss_fn=loss_fn, collect=True
            )
            if val_n == 0:
                val_loss, val_acc = train_loss, train_acc
                y_true, y_pred = [], []
        else:
            val_loss, val_acc = train_loss, train_acc
            y_true, y_pred = [], []

        balanced = _balanced_accuracy(y_true, y_pred) if y_true else 0.0
        macro_f1 = _macro_f1(y_true, y_pred) if y_true else 0.0
        score, _ = _selection_metric(early_metric, val_loss, val_acc, y_true, y_pred)

        learning_rate = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(learning_rate)
        history["val_balanced_accuracy"].append(float(balanced))
        history["val_macro_f1"].append(float(macro_f1))

        improved = (
            score > best_score + min_delta
            if higher_is_better
            else score < best_score - min_delta
        )
        if improved:
            best_score = float(score)
            best_val_loss = float(val_loss)
            best_val_acc = float(val_acc)
            best_epoch = epoch
            wait = 0
            checkpoint_extra = {
                "early_metric": early_metric,
                "best_val_metric": best_score,
                "val_balanced_accuracy": float(balanced),
                "val_macro_f1": float(macro_f1),
                "loss": loss_name,
                "class_weight": class_weight,
                "class_weights_source": class_weights_source,
                "class_weights": None if weights is None else [float(v) for v in weights],
                "train_class_counts": train_class_counts,
                "balanced_sampler": bool(_cfg_get(cfg, "balanced_sampler", False)),
            }
            _save_checkpoint(
                model, cfg, epoch, best_val_loss, best_val_acc, checkpoint_file,
                extra=checkpoint_extra,
            )
        else:
            wait += 1

        scheduler.step()
        if wait >= patience:
            early_stopped = True
            break

    if best_epoch < 0:  # degenerate losses (e.g. all NaN): still emit an artifact
        best_epoch = max(0, len(history["train_loss"]) - 1)
        best_val_loss = float(history["val_loss"][-1]) if history["val_loss"] else None
        best_val_acc = float(history["val_acc"][-1]) if history["val_acc"] else None
        best_score = (
            float(history["val_balanced_accuracy"][-1])
            if early_metric == "balanced_accuracy" and history["val_balanced_accuracy"]
            else (
                float(history["val_macro_f1"][-1])
                if early_metric == "macro_f1" and history["val_macro_f1"]
                else best_val_loss
            )
        )
        _save_checkpoint(
            model, cfg, best_epoch, best_val_loss, best_val_acc, checkpoint_file,
            extra={
                "early_metric": early_metric,
                "best_val_metric": best_score,
                "loss": loss_name,
                "class_weight": class_weight,
                "class_weights_source": class_weights_source,
                "class_weights": None if weights is None else [float(v) for v in weights],
                "train_class_counts": train_class_counts,
                "balanced_sampler": bool(_cfg_get(cfg, "balanced_sampler", False)),
            },
        )

    _write_json(history_file, history)
    return {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
        "best_val_metric": best_score,
        "early_metric": early_metric,
        "loss": loss_name,
        "class_weight": class_weight,
        "class_weights_source": class_weights_source,
        "class_weights": None if weights is None else [float(v) for v in weights],
        "train_class_counts": train_class_counts,
        "balanced_sampler": bool(_cfg_get(cfg, "balanced_sampler", False)),
        "epochs_run": len(history["train_loss"]),
        "early_stopped": bool(early_stopped),
        "checkpoint_path": str(checkpoint_file),
        "history_path": str(history_file),
        "config": _config_snapshot(cfg),
        "commit_hash": _git_commit_hash(),
        "optimizer": str(_cfg_get(cfg, "optimizer", "adamw")),
    }


def evaluate(model: Any, loader: Any, cfg: Any) -> dict:
    """Evaluate ``model`` on ``loader`` and return JSON-native binary metrics.

    All numbers come from :mod:`experiments.metrics` (``binary_metrics``,
    ``log_loss`` and ``calibration_summary``).  Softmax probabilities from the
    model are supplied as ``y_prob`` so the AUC, log loss and calibration fields
    are computed from the same score vector used for the hard predictions.
    """
    _require_torch()
    n_bins = int(_cfg_get(cfg, "calibration_bins", 10))
    device, dtype = _model_device_and_dtype(model)
    y_true: list[int] = []
    probabilities: list[float] = []
    predictions: list[int] = []
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                X, y = _prepare_batch(batch, device, dtype)
                logits = model(X)
                _, probability, prediction = _binary_outputs(logits, y)
                y_true.extend(int(v) for v in y.detach().cpu().numpy().tolist())
                probabilities.extend(float(v) for v in probability.detach().cpu().numpy().tolist())
                predictions.extend(int(v) for v in prediction.detach().cpu().numpy().tolist())
    finally:
        model.train(was_training)

    if not y_true:
        return {
            "n_samples": 0,
            "support": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "auc": None,
            "iou_class0": None,
            "iou_class1": None,
            "macro_iou": None,
            "confusion_matrix": [],
            "log_loss": None,
            "calibration": {"status": "N/A", "reason": "empty loader"},
        }

    from experiments import metrics as _metrics

    result = _metrics.binary_metrics(y_true, predictions, y_prob=probabilities)
    calibration = _metrics.calibration_summary(y_true, probabilities, n_bins=n_bins)
    return {
        "n_samples": int(result.support),
        "support": int(result.support),
        "accuracy": float(result.accuracy),
        "precision": float(result.precision),
        "recall": float(result.recall),
        "f1": float(result.f1),
        "auc": None if result.auc is None else float(result.auc),
        "iou_class0": float(result.iou_class0),
        "iou_class1": float(result.iou_class1),
        "macro_iou": float(result.macro_iou),
        "confusion_matrix": result.confusion.tolist(),
        "log_loss": _metrics.log_loss(y_true, probabilities),
        "calibration": {
            "brier_score": calibration["brier_score"],
            "ece": calibration["ece"],
            "n_bins": calibration["n_bins"],
            "n_samples": calibration["n_samples"],
        },
    }
