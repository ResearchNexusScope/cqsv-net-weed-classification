"""Training and evaluation utilities for the rebuilt hybrid QNN pipeline (QML-D02).

The module is deliberately configuration- and data-agnostic: a caller supplies a
model, two DataLoaders and a config object exposing the QMLConfig fields listed in
the swarm contract.  No dataset path, no paper number and no PennyLane import is
present here, so the module stays importable in the central verifier's
Python/PennyLane-free environment.  Quantum models are constructed by QML-D01's
``qmlpipe.models`` builders and passed in by the CLI.

Loss/accuracy are computed from the two-way logits emitted by every model in
``models/quantum.py`` (and by a standard ``nn.Linear(n, 2)`` head).  A single
logit per sample is also accepted via the sigmoid/BCE path.  Every metric is
delegated to :mod:`experiments.metrics`; this module embeds no numeric results.
"""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments import metrics as _metrics

try:  # pragma: no cover - a minimal verifier may not install torch
    import torch
    import torch.nn.functional as F

    _TORCH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

__all__ = ["seed_everything", "train", "evaluate"]

HISTORY_KEYS = ("train_loss", "val_loss", "train_acc", "val_acc", "lr")
_SUPPORTED_OPTIMIZERS = ("adamw", "adam", "sgd", "spsa")


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


def _binary_outputs(logits: Any, y: Any):
    """Return ``(loss, positive_probability, hard_prediction)`` for binary logits.

    Accepts a two-column logit tensor, a single logit column, a one-dimensional
    logit vector, or a tuple/list whose first element is a logit tensor.
    """
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits, dtype=torch.float32)
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim == 1:
        float_logits = logits.float()
        probability = torch.sigmoid(float_logits)
        prediction = (probability >= 0.5).long()
        loss = F.binary_cross_entropy_with_logits(float_logits, y.float())
    elif logits.ndim == 2 and logits.shape[1] >= 2:
        two_logits = logits[:, :2]
        probability = torch.softmax(two_logits.float(), dim=1)[:, 1]
        prediction = torch.argmax(two_logits, dim=1)
        loss = F.cross_entropy(two_logits, y)
    else:
        raise ValueError(f"Unsupported logits shape {tuple(logits.shape)} for binary training")
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


def _train_step(model: Any, X: Any, y: Any, optimizer: Any, cfg: Any, is_spsa: bool):
    """One optimisation step; returns ``(loss_value, predictions)``."""
    if is_spsa:
        # SPSA estimates the gradient from two loss evaluations and therefore
        # takes a closure that returns the loss without calling backward().
        observed: list[Any] = []

        def closure():
            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss, _, prediction = _binary_outputs(logits, y)
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
            _, _, prediction = _binary_outputs(model(X), y)
        return loss_value, prediction.detach()

    optimizer.zero_grad(set_to_none=True)
    logits = model(X)
    loss, _, prediction = _binary_outputs(logits, y)
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
):
    """Mean loss and accuracy over one pass of ``loader`` (train or eval)."""
    _require_torch()
    device, dtype = _model_device_and_dtype(model)
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            X, y = _prepare_batch(batch, device, dtype)
            if training:
                loss_value, prediction = _train_step(model, X, y, optimizer, cfg, is_spsa)
            else:
                logits = model(X)
                loss, _, prediction = _binary_outputs(logits, y)
                loss_value = float(loss.detach().cpu())
            batch_n = int(y.shape[0])
            total_loss += loss_value * batch_n
            total_correct += int((prediction == y).sum().item())
            total_n += batch_n
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
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


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
    """Train ``model`` with early stopping and persist history + best checkpoint.

    Optimizer, learning rate, weight decay, patience and epochs all come from
    ``cfg``.  A full cosine schedule is stepped once per epoch.  Gradients are
    clipped by ``cfg.grad_clip`` (default 1.0) for gradient-based optimizers;
    SPSA uses its closure API and has no backpropagated gradient to clip.

    Returns a JSON-serialisable dictionary containing the full ``history``
    (keys: ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``, ``lr``),
    the best epoch/validation metrics, the number of epochs actually run, and
    whether early stopping fired.  The history is written to ``history.json``
    under ``output_dir`` (or ``history_path``) and the best checkpoint to
    ``checkpoints/best.pt`` (or ``ckpt_path``).
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

    optimizer, is_spsa = _build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: dict[str, list[float]] = {key: [] for key in HISTORY_KEYS}
    best_loss = float("inf")
    best_epoch = -1
    best_val_acc: float | None = None
    wait = 0
    early_stopped = False

    for epoch in range(epochs):
        train_loss, train_acc, _ = _epoch_loss_acc(
            model, train_loader, cfg, optimizer=optimizer, is_spsa=is_spsa
        )
        if val_loader is not None:
            val_loss, val_acc, val_n = _epoch_loss_acc(model, val_loader, cfg)
            if val_n == 0:
                val_loss, val_acc = train_loss, train_acc
        else:
            val_loss, val_acc = train_loss, train_acc

        learning_rate = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(learning_rate)

        if val_loss < best_loss:
            best_loss = float(val_loss)
            best_val_acc = float(val_acc)
            best_epoch = epoch
            wait = 0
            _save_checkpoint(model, cfg, epoch, best_loss, best_val_acc, checkpoint_file)
        else:
            wait += 1

        scheduler.step()
        if wait >= patience:
            early_stopped = True
            break

    if best_epoch < 0:  # degenerate losses (e.g. all NaN): still emit an artifact
        best_epoch = len(history["train_loss"]) - 1
        best_loss = float(history["val_loss"][-1]) if history["val_loss"] else None
        best_val_acc = float(history["val_acc"][-1]) if history["val_acc"] else None
        _save_checkpoint(model, cfg, best_epoch, best_loss, best_val_acc, checkpoint_file)

    _write_json(history_file, history)
    return {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": best_loss,
        "best_val_accuracy": best_val_acc,
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
