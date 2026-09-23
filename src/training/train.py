"""
train.py

Phase 1, step 2 (training): trains BindingSiteGNN on the assembled
graphs.pt dataset, with class-imbalance-aware loss and metrics.

Why NOT plain accuracy: the real class balance found in the pilot
dataset is ~2-3% positive (binding-site) residues across all splits.
A model predicting "not a binding site" for every residue would score
~97-98% accuracy while being useless. This script:
  - uses BCEWithLogitsLoss with pos_weight computed from the actual
    training-set class ratio, so the rare positive class is weighted
    proportionally in the loss
  - reports precision, recall, F1, and AUROC on the positive
    (binding-site) class specifically, every epoch, on train AND val
  - saves the checkpoint with the best val F1, not best val loss
    (loss can improve while F1 stays flat/degrades under imbalance)

This is a smoke test on 15 proteins (10 train / 3 val / 2 test) — the
goal is confirming the pipeline learns SOMETHING coherent, not
producing a publishable result. Treat val/test metrics as noisy at
this sample size.

Run from the project root:
    python src/training/train.py
    python src/training/train.py --epochs 100 --lr 0.001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.encoder import BindingSiteGNN  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPHS_PATH = PROJECT_ROOT / "data" / "processed" / "graphs.pt"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "processed" / "best_model.pt"


def compute_pos_weight(graphs: list) -> torch.Tensor:
    """pos_weight for BCEWithLogitsLoss = n_negative / n_positive, computed
    from the TRAINING set only (val/test must never influence the loss
    or any other training decision — same principle as not augmenting
    test data, discussed earlier)."""
    total_pos = sum(g.y.sum().item() for g in graphs)
    total = sum(g.y.numel() for g in graphs)
    total_neg = total - total_pos
    if total_pos == 0:
        print("WARNING: zero positive examples in training set — pos_weight "
              "undefined, defaulting to 1.0 (loss will not be imbalance-aware)")
        return torch.tensor(1.0)
    return torch.tensor(total_neg / total_pos)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        all_logits.append(logits.cpu())
        all_labels.append(batch.y.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.numpy(), preds.numpy(), average="binary", zero_division=0
    )
    try:
        auroc = roc_auc_score(labels.numpy(), probs.numpy())
    except ValueError:
        # happens if a split has only one class present (e.g. all-negative)
        auroc = float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "n_positive": int(labels.sum().item()),
        "n_total": int(labels.numel()),
    }


def format_metrics(metrics: dict) -> str:
    return (
        f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
        f"F1={metrics['f1']:.3f} AUROC={metrics['auroc']:.3f} "
        f"(pos={metrics['n_positive']}/{metrics['n_total']})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                         help="L2 regularization strength (Adam weight_decay)")
    parser.add_argument("--patience", type=int, default=6,
                         help="stop if val F1 hasn't improved for this many "
                              "VALIDATION CHECKS in a row (checks happen every "
                              "5 epochs, so patience=6 means ~30 epochs without "
                              "improvement before stopping)")
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if not GRAPHS_PATH.exists():
        print(f"ERROR: {GRAPHS_PATH} not found. Run build_dataset.py first.")
        sys.exit(1)

    data = torch.load(GRAPHS_PATH, weights_only=False)
    train_graphs, val_graphs, test_graphs = data["train"], data["val"], data["test"]

    if len(train_graphs) == 0:
        print("ERROR: training set is empty. Nothing to train on.")
        sys.exit(1)
    if len(val_graphs) == 0:
        print("WARNING: validation set is empty — best-checkpoint selection "
              "and early stopping will not be meaningful.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Train/Val/Test proteins: {len(train_graphs)}/{len(val_graphs)}/{len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    pos_weight = compute_pos_weight(train_graphs).to(device)
    print(f"pos_weight (neg/pos ratio in train): {pos_weight.item():.2f}")

    model = BindingSiteGNN(
        in_channels=train_graphs[0].x.shape[1],
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0  # counted in VALIDATION CHECKS, not raw epochs
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)

        if epoch % 5 == 0 or epoch == args.epochs:
            train_metrics = evaluate(model, train_loader, device)
            print(f"\nEpoch {epoch:3d} | loss={avg_loss:.4f}")
            print(f"  train: {format_metrics(train_metrics)}")

            if len(val_graphs) > 0:
                val_metrics = evaluate(model, val_loader, device)
                print(f"  val:   {format_metrics(val_metrics)}")

                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), CHECKPOINT_PATH)
                    print(f"  -> new best val F1 ({best_val_f1:.3f}), checkpoint saved")
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= args.patience:
                        print(
                            f"\nEarly stopping: val F1 has not improved for "
                            f"{args.patience} validation checks "
                            f"(since epoch {best_epoch}, best F1={best_val_f1:.3f})"
                        )
                        stopped_early = True
                        break

    print("\n" + "=" * 50)
    print("Training complete" + (" (stopped early)" if stopped_early else ""))
    print("=" * 50)

    if len(val_graphs) > 0 and best_epoch > 0:
        print(f"Best val F1: {best_val_f1:.3f} at epoch {best_epoch}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    else:
        print("No validation-based checkpoint selection occurred "
              "(empty val set) — using final-epoch weights for test evaluation.")

    if len(test_graphs) > 0:
        test_metrics = evaluate(model, test_loader, device)
        print(f"Test: {format_metrics(test_metrics)}")
        print(
            "\nNOTE: test set has only "
            f"{len(test_graphs)} protein(s) — treat this number as a "
            "sanity check, not a statistically meaningful result."
        )
    else:
        print("Test set is empty — skipping test evaluation.")


if __name__ == "__main__":
    main()
