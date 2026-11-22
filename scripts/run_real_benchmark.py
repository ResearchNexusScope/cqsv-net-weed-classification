"""Real-data benchmark driver for the rebuilt Quantum_Classification pipeline.

Uses the public Kaggle CornWeed dataset and the DatasetNinja mirror. All data
paths come from environment variables; nothing is hardcoded to a developer
machine. OOD crops are capped by SOURCE IMAGE group so a CPU-only run is
feasible and leak-free.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import replace
import tempfile
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from torchvision import models, transforms

from data import build_splits
from data.manifest import SplitRecord
from data.splits import Split
from experiments import runner
from experiments.config import ExperimentConfig
from experiments.evaluate import CallablePredictor


class SubsetManifest:
    """Minimal manifest view with a capped OOD split (group-disjoint)."""

    def __init__(self, base, ood_samples):
        self.splits = dict(base.splits)
        self.splits[Split.OOD_TEST] = SplitRecord(
            name=Split.OOD_TEST.value, samples=tuple(ood_samples)
        )
        self.config_snapshot = getattr(base, "config_snapshot", {})
        self.seed = getattr(base, "seed", 42)

    def _to_payload(self):
        return {}


def prepare_ninja_subset(n_files: int = 1200):
    """Hard-link a small, source-disjoint Ninja subset into temp dirs.

    The full mirror contains 46k images but only ~1k carry box annotations.
    Linking the first N annotated source images makes build_splits fast while
    preserving the original annotation/image filename contract.
    """
    full_ann = Path(os.environ["NINJA_ANN_DIR"])
    full_img = Path(os.environ["NINJA_IMG_DIR"])
    tmp = Path(tempfile.mkdtemp(prefix="qc_ninja_subset_"))
    ann_out, img_out = tmp / "ann", tmp / "img"
    ann_out.mkdir()
    img_out.mkdir()

    def link(src: Path, dst: Path):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    linked = 0
    for ann_file in sorted(full_ann.glob("*.json")):
        if linked >= n_files:
            break
        image_name = Path(ann_file).stem  # e.g. C00020598.jpg
        image_file = full_img / Path(image_name).name
        if not image_file.exists():
            continue
        link(ann_file, ann_out / ann_file.name)
        link(image_file, img_out / image_file.name)
        linked += 1
    if linked == 0:
        raise RuntimeError("no annotated Ninja source images found")
    os.environ["NINJA_ANN_DIR"] = str(ann_out)
    os.environ["NINJA_IMG_DIR"] = str(img_out)
    print(f"ninja subset: linked {linked} source images -> {tmp}")
    return tmp


def cap_ood_by_group(samples, per_class: int):
    """Pick whole source-image groups until each class reaches per_class crops."""
    groups = defaultdict(list)
    for sample in samples:
        groups[(sample.group_id, sample.label)].append(sample)
    keys = sorted(groups)
    chosen = {0: [], 1: []}
    for key in keys:
        _, label = key
        if len(chosen[label]) >= per_class:
            continue
        chosen[label].extend(groups[key])
    out = chosen[0][:per_class] + chosen[1][:per_class]
    return out


def build_backbone(device: torch.device):
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.eval().to(device)
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


def make_feature_loader(backbone, preprocess, device):
    cache = {}

    def loader(samples, cfg, split):
        missing = []
        for sample in samples:
            key = (sample.sample_id, tuple(sample.bbox or ()))
            if key not in cache:
                missing.append((key, sample))
        if missing:
            tensors = []
            for _, sample in missing:
                image = Image.open(sample.path).convert("RGB")
                if getattr(sample, "bbox", None):
                    x1, y1, x2, y2 = [int(v) for v in sample.bbox]
                    image = image.crop((x1, y1, x2, y2))
                tensors.append(preprocess(image))
            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                features = backbone.features(batch)
                features = backbone.avgpool(features).flatten(1)
                features = F.normalize(features, p=2, dim=1)
            arrays = features.cpu().numpy().astype(np.float32)
            for (key, _), array in zip(missing, arrays):
                cache[key] = array
        return np.stack([cache[(s.sample_id, tuple(s.bbox or ()))] for s in samples])
    return loader


def make_fitted_baseline(X_fit, y_fit):
    """A classical logistic baseline fitted once on real Kaggle train features."""
    clf = LogisticRegression(max_iter=2000, random_state=42).fit(X_fit, y_fit)

    def predict(X, fit_X, fit_y, seed):
        start = time.perf_counter()
        y_pred = clf.predict(X)
        score_pos = clf.predict_proba(X)[:, 1]
        ms = (time.perf_counter() - start) * 1000.0 / max(1, len(X))
        return y_pred, score_pos, ms

    return CallablePredictor(predict)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/real_run")
    parser.add_argument("--oor-per-class", type=int, default=200)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    kaggle_root = os.environ["KAGGLE_DATA_DIR"]
    prepare_ninja_subset()
    cfg = replace(ExperimentConfig.from_env(), output_dir=args.output_dir)

    t0 = time.time()
    base_manifest = build_splits(cfg, kaggle_root=kaggle_root)
    ood = cap_ood_by_group(base_manifest.splits[Split.OOD_TEST].samples, args.oor_per_class)
    manifest = SubsetManifest(base_manifest, ood)
    print(f"manifest built in {time.time()-t0:.1f}s; ood test crops={len(ood)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, preprocess = build_backbone(device)
    feature_loader = make_feature_loader(backbone, preprocess, device)

    # Pre-train the classical baseline once on real Kaggle training features.
    train_samples = manifest.splits[Split.TRAIN].samples
    X_train = feature_loader(train_samples, cfg, "train")
    y_train = np.asarray([sample.label for sample in train_samples], dtype=int)
    baseline = make_fitted_baseline(X_train, y_train)
    print(f"classical baseline fitted on {len(y_train)} real training crops")

    noise_experiment = getattr(runner, "NOISE_EXPERIMENT", "noise_robustness")
    spec = runner.ExperimentSpec(
        experiment_id="real_ood_benchmark",
        experiments=(runner.MAIN_EXPERIMENT, runner.QSVC_EXPERIMENT, noise_experiment),
        manifest=manifest,
        feature_loader=feature_loader,
        data_provider=runner.manifest_data_provider,
        model_ids=("logreg_effnet", "svc_rbf", "svc_poly2_classical", "qsvc_quantum_kernel", "qsvc_no_mixing_control"),
        model_factories={"logreg_effnet": lambda cfg: baseline},
        seeds_per_model=args.runs,
        noise_draws=1,
        sigma_grid=(0.0, 0.05, 0.10, 0.20),
        bootstrap_n=200,
        comparison_pairs=(("qsvc_quantum_kernel", "svc_poly2_classical"),),
    )
    resolved = runner.resolve_config(
        ["run", "--output-dir", args.output_dir, "--runs", str(args.runs), "--noise-draws", "1"],
        config=cfg,
        spec=spec,
    )
    print("running real benchmark...")
    manifest_result = runner.run_experiment(resolved, spec)
    print("done:", manifest_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
