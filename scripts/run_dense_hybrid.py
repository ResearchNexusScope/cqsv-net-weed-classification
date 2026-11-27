"""Train the existing CQSV-Net hybrid and compare its locked predictions on DeepWeeds."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import fresh_protocol as fp
from qmlpipe import fresh_hybrid as hybrid


def demo_features(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(96, 512)).astype("float32")
    y = (x[:, 0] + x[:, 1] > 0).astype("int64")
    return {"X": {"train": x[:64], "val": x[64:]},
            "y": {"train": y[:64], "val": y[64:]}}


def split_digest(data):
    """Bind a trained checkpoint to the exact row identities and frozen features."""
    digest = hashlib.sha256()
    for split in ("train", "val", "test"):
        ids = json.dumps(data["sample_ids"][split], separators=(",", ":")).encode()
        digest.update(split.encode())
        digest.update(ids)
        digest.update(np.ascontiguousarray(data["X"][split], dtype=np.float32).view(np.uint8))
        if split != "test":
            digest.update(np.ascontiguousarray(data["y"][split], dtype=np.int64).view(np.uint8))
    return digest.hexdigest()


def paired_rbf(x_train, y_train, x_eval, truth, predicted, seed):
    rbf = fp._make_estimator("rbf", seed)
    rbf.fit(x_train, y_train)
    baseline = rbf.predict(x_eval)
    correct, baseline_correct = predicted == truth, baseline == truth
    return baseline, dict(accuracy=float(baseline_correct.mean()),
                          fixed=int(np.sum(correct & ~baseline_correct)),
                          broken=int(np.sum(~correct & baseline_correct)),
                          difference_points=float(100 * (correct.mean() - baseline_correct.mean())))


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch installation cannot access a GPU.")
    final = args.mode == "final"
    demo = args.mode == "demo"
    if final:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("dataset") != "DeepWeeds" or checkpoint["train_cap"]["cap_applied"]:
            raise ValueError("Final evaluation requires a full-training DeepWeeds validation checkpoint.")
        cfg = dict(checkpoint["config"], torch_device=args.device)
    else:
        cfg = dict(qubits=args.qubits, layers=args.layers, epochs=args.epochs, seed=args.seed,
                   split_seed=args.seed if args.split_seed is None else args.split_seed,
                   embedding=args.encoding, batch_size=args.batch_size, max_batches_per_epoch=0,
                   validation_patience=args.patience, torch_device=args.device,
                   arm=args.arm, log_epochs=True)
    if cfg["embedding"] == "amplitude" and (device.type != "cpu" or cfg["arm"] != "hybrid" or cfg["qubits"] > 10):
        raise ValueError("The legacy amplitude comparison supports CPU with at most 10 qubits.")
    seed = cfg["seed"]
    data = demo_features(seed) if demo else fp.cache_features(
        "deepweeds", "plantnet", root=args.root, cache_dir=args.cache_dir, seed=cfg["split_seed"])
    digest = None if demo else split_digest(data)
    x, y = data["X"], data["y"]
    if final and checkpoint["split_digest"] != digest:
        raise ValueError("The sample IDs, frozen features or training/validation labels have changed.")
    train_x, train_y, cap = fp.cap_train(x["train"], y["train"], args.train_cap, seed)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
    if final:
        model = hybrid.build_hybrid_model(cfg, checkpoint["feature_dim"]).to(device)
        model.load_state_dict(checkpoint["weights"])
        history = checkpoint["history"]
    else:
        model, history = hybrid.train_hybrid_from_features(
            train_x, train_y, x["val"], y["val"], cfg, seed)
    split = "test" if final else "val"
    logits = hybrid._predict_logits(model, x[split], cfg["batch_size"])
    predictions = logits.argmax(axis=1)
    report = dict(mode=args.mode, dataset="synthetic fixture" if demo else "DeepWeeds",
                  backbone="synthetic 512D features" if demo else "PlantNet", task="binary",
                  config=cfg, train_cap=cap, history=history, split=split,
                  torch_version=str(torch.__version__), n=len(predictions),
                  accuracy=float(np.mean(predictions == y[split])), split_digest=digest,
                  cache_path=data.get("cache_path"), split_provenance=data.get("provenance"))
    if demo and cfg["arm"] == "hybrid":
        for key in ("adapter_grad_norm", "circuit_grad_norm"):
            if not np.isfinite(history[key]) or history[key] <= 0:
                raise RuntimeError(f"Expected a positive finite {key} in the demo")
    if not demo:
        baseline, report["paired_plantnet_rbf"] = paired_rbf(
            train_x, train_y, x[split], y[split], predictions, seed)
        np.savez_compressed(args.output / "paired_predictions.npz", sample_ids=data["sample_ids"][split],
                            labels=y[split], model=predictions, rbf=baseline, logits=logits)
    if args.output:
        if not final:
            torch.save({"config": cfg, "dataset": report["dataset"], "feature_dim": int(train_x.shape[1]),
                        "weights": model.state_dict(), "split_digest": digest, "train_cap": cap,
                        "history": history}, args.output / "best.pt")
        else:
            report["checkpoint"] = str(args.checkpoint)
        (args.output / "metrics.json").write_text(
            json.dumps(fp._json_safe(report), indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"history", "split_provenance"}}))
    print(json.dumps({k: history[k] for k in ("selected_epoch", "selected_val_loss", "adapter_grad_norm",
                                              "circuit_grad_norm", "parameter_counts")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("demo", "validate", "final"):
        sub = modes.add_parser(mode)
        sub.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        sub.add_argument("--output", type=Path, required=mode != "demo", help="new results directory")
        if mode != "demo":
            sub.add_argument("--root", help="dataset root used by the existing DeepWeeds loader")
            sub.add_argument("--cache-dir", help="frozen feature cache directory")
        if mode == "final":
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.set_defaults(train_cap=0)
            continue
        sub.add_argument("--qubits", type=int, choices=range(4, 21), default=5)
        sub.add_argument("--encoding", choices=["amplitude", "dense_angle"], default="dense_angle")
        sub.add_argument("--arm", choices=["hybrid", "classical", "replacement"], default="hybrid")
        sub.add_argument("--epochs", type=int, default=2 if mode == "demo" else 30)
        sub.add_argument("--layers", type=int, default=2)
        sub.add_argument("--batch-size", type=int, default=8)
        sub.add_argument("--patience", type=int, default=5)
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--split-seed", type=int, help="group split seed; defaults to --seed")
        sub.add_argument("--train-cap", type=int, default=0, help="0 uses every training row")
    args = parser.parse_args()
    if args.mode != "final" and (min(args.epochs, args.layers, args.batch_size, args.patience) < 1 or args.train_cap < 0):
        parser.error("Epochs, layers, batch size and patience must be positive and cap must be >= 0.")
    run(args)


if __name__ == "__main__":
    main()
