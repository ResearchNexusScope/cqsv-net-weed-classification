"""Synthetic CLI tests for frozen-backbone selection (PLANT-D02).

All fixtures are in-memory: no real dataset path, no Hugging Face download, no
paper result number.  Tests that would need torch/torchvision/Hugging Face or
sklearn skip cleanly so this module stays importable under the mother's minimal
verifier environment.  Heavy imports inside the CLIs are lazy, so the dry-run
tests exercise the full argument/plan path without loading a model.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These two modules import only the standard library at module scope.
from scripts import compare_backbones, train_qml  # noqa: E402


# ---------------------------------------------------------------------------
# train_qml CLI / plan
# ---------------------------------------------------------------------------
def test_train_parser_exposes_backbone() -> None:
    parser = train_qml._build_parser()
    empty = parser.parse_args([])
    assert hasattr(empty, "backbone")
    assert empty.backbone is None  # plan-level default is applied in _resolve_plan
    assert parser.parse_args(["--backbone", "imagenet"]).backbone == "imagenet"


def test_train_parser_rejects_invalid_backbone() -> None:
    with pytest.raises(SystemExit) as exc:
        train_qml._build_parser().parse_args(["--backbone", "resnet50"])
    assert exc.value.code == 2


def test_train_plan_default_backbone_and_width() -> None:
    plan, _ = train_qml._resolve_plan(train_qml._build_parser().parse_args([]))
    assert plan["backbone"] == "plantnet"
    assert plan["backbone"] in train_qml.ALLOWED_BACKBONES
    # classical_dim tracks the selected backbone's raw feature width so the
    # feature-fed QNN projection consumes correctly shaped vectors.
    assert plan["classical_dim"] == train_qml.BACKBONE_FEATURE_DIM["plantnet"]


def test_train_plan_imagenet_backbone_and_width() -> None:
    plan, _ = train_qml._resolve_plan(
        train_qml._build_parser().parse_args(["--backbone", "imagenet"])
    )
    assert plan["backbone"] == "imagenet"
    assert plan["classical_dim"] == train_qml.BACKBONE_FEATURE_DIM["imagenet"]


def test_train_plan_config_json_backbone(tmp_path: Path) -> None:
    cfg = tmp_path / "backbone.json"
    cfg.write_text(json.dumps({"backbone": "imagenet"}), encoding="utf-8")
    plan, _ = train_qml._resolve_plan(
        train_qml._build_parser().parse_args(["--config-json", str(cfg)])
    )
    assert plan["backbone"] == "imagenet"


def test_train_main_rejects_invalid_backbone_from_config_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "bad_backbone.json"
    cfg.write_text(json.dumps({"backbone": "bogus"}), encoding="utf-8")
    target = tmp_path / "would_be_output"
    code = train_qml.main(
        ["--dry-run", "--config-json", str(cfg), "--output-dir", str(target)]
    )
    assert code == 2
    assert "backbone" in capsys.readouterr().err
    assert not target.exists()


def test_train_dry_run_exposes_backbone_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "dry_run_output"
    code = train_qml.main(
        ["--dry-run", "--backbone", "plantnet", "--output-dir", str(target)]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "dry_run"
    assert payload["plan"]["backbone"] == "plantnet"
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# provenance metadata
# ---------------------------------------------------------------------------
def test_backbone_provenance_plantnet(monkeypatch: pytest.MonkeyPatch) -> None:
    """PlantNet repo/weights/license are recorded without importing real torch."""
    fake = types.ModuleType("qmlpipe.plant_backbone")
    fake.PLANTNET_REPO_ID = "cpoisson/plantnet300k-resnet18"
    fake.PLANTNET_WEIGHTS_FILE = "plantnet_resnet18.pth"
    monkeypatch.setitem(sys.modules, "qmlpipe.plant_backbone", fake)
    info = train_qml._backbone_provenance("plantnet", 512)
    assert info["name"] == "plantnet"
    assert info["feature_dim"] == 512
    assert info["frozen"] is True
    assert info["repo_id"] == "cpoisson/plantnet300k-resnet18"
    assert info["weights_file"] == "plantnet_resnet18.pth"
    assert info["license"] == "Apache-2.0"


def test_backbone_provenance_imagenet() -> None:
    info = train_qml._backbone_provenance("imagenet", 1280)
    assert info["name"] == "imagenet"
    assert info["feature_dim"] == 1280
    assert info["frozen"] is True
    assert "repo_id" not in info  # PlantNet-specific metadata is not attached


# ---------------------------------------------------------------------------
# feature-extractor dispatch (needs torch + torchvision; skip cleanly)
# ---------------------------------------------------------------------------
def test_call_feature_extractor_both_branches_are_l2_normalised() -> None:
    """Both dispatch paths yield unit-L2 features on a synthetic batch.

    ``Direct`` mimics the frozen PlantNet module (no ``.features``/``.avgpool``;
    ``forward`` returns L2-normalised vectors).  ``Trunked`` mimics the legacy
    EfficientNet path that exposes ``.features``/``.avgpool``.  No Hugging Face
    download or real backbone is involved.
    """
    torch = pytest.importorskip("torch", reason="torch is required")
    from torch import nn
    from torch.nn import functional as F

    from scripts.run_real_benchmark import call_feature_extractor

    class Direct(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.proj = nn.Linear(3 * 4 * 4, dim)

        def forward(self, x):
            return F.normalize(self.proj(x.flatten(1)), p=2, dim=1)

    class Trunked(nn.Module):
        """Mimics the legacy EfficientNet path: .features + .avgpool only.

        ``call_feature_extractor``'s trunked branch mirrors the old runner
        (``features -> avgpool -> flatten -> F.normalize``); it applies no
        projection head here, so the output width equals the avgpool channel
        count, not any classifier dim.
        """

        def __init__(self, channels: int = 4) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, channels, 3, padding=1), nn.ReLU()
            )
            self.avgpool = nn.AdaptiveAvgPool2d(1)

    batch = torch.randn(2, 3, 4, 4)
    with torch.no_grad():
        direct_out = call_feature_extractor(Direct(5), batch)
        trunked_out = call_feature_extractor(Trunked(4), batch)
    # Direct (PlantNet-style) branch returns its own projected width; the
    # trunked (ImageNet-style) branch returns the avgpool channel count.
    for features, expected_dim in ((direct_out, 5), (trunked_out, 4)):
        assert tuple(features.shape) == (2, expected_dim)
        norms = features.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# compare_backbones CLI
# ---------------------------------------------------------------------------
def test_compare_parser_exposes_backbone() -> None:
    parser = compare_backbones._build_parser()
    assert parser.parse_args([]).backbone == "plantnet"
    assert parser.parse_args(["--backbone", "imagenet"]).backbone == "imagenet"


def test_compare_parser_rejects_invalid_backbone() -> None:
    with pytest.raises(SystemExit) as exc:
        compare_backbones._build_parser().parse_args(["--backbone", "resnet50"])
    assert exc.value.code == 2


def test_compare_resolve_plan_rejects_invalid_backbone() -> None:
    args = types.SimpleNamespace(
        backbone="bogus",
        output_dir="results/x",
        subset_files=8,
        oor_per_class=2,
        seed=42,
    )
    with pytest.raises(ValueError, match="backbone"):
        compare_backbones._resolve_plan(args)


def test_compare_dry_run_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "compare_dry"
    code = compare_backbones.main(["--dry-run", "--output-dir", str(target)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "dry_run"
    assert payload["plan"]["backbone"] == "plantnet"
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_compare_fit_uses_only_train_features(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fitted scaler sees only TRAIN features, never the held-out split."""
    np = pytest.importorskip("numpy", reason="numpy is required")
    pytest.importorskip("sklearn", reason="scikit-learn is required")
    from sklearn.preprocessing import StandardScaler

    seen: dict[str, np.ndarray] = {}
    original_fit = StandardScaler.fit

    def spy(self, X, *args, **kwargs):
        seen["X"] = np.asarray(X).copy()
        return original_fit(self, X, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", spy)

    X_train = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.full((3, 2), 50.0)  # clearly distinct from training features
    y_test = np.array([0, 1, 1])

    scores = compare_backbones.fit_and_evaluate(
        X_train, y_train, {"ood_test": (X_test, y_test)}, seed=0
    )

    assert "X" in seen
    assert np.array_equal(seen["X"], X_train)
    assert not np.array_equal(seen["X"], X_test)
    # Reported metrics are computed, not hardcoded.
    for entry in scores.values():
        assert 0.0 <= entry["accuracy"] <= 1.0
        assert entry["n"] > 0
        assert "auc" in entry
