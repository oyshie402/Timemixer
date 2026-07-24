

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from timemixer import load_etth1, run_epoch, SeriesDecomp

torch.manual_seed(2000)
np.random.seed(2000)


class DLinear(nn.Module):
    
    def __init__(self, seq_len: int, pred_len: int, decomp_kernel: int = 25):
        super().__init__()
        self.decomp = SeriesDecomp(decomp_kernel)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor):
        # x: [B, seq_len, C]
        seasonal, trend = self.decomp(x)
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        trend = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        return seasonal + trend  # [B, pred_len, C]


def train_dlinear(pred_len, csv_path="ETTh1.csv", seq_len=96,
                   batch_size=128, epochs=20, lr=1e-2, patience=10, verbose=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, test_ds, (mean, std), c_in, cols = load_etth1(csv_path, seq_len, pred_len)
    tag = f"dlinear_pred{pred_len}"
    if verbose:
        print(f"[{tag}] train/val/test windows: "
              f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)} | variates: {c_in} | device: {device}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = DLinear(seq_len=seq_len, pred_len=pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_mae = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = run_epoch(model, val_loader, criterion, None, device)
        train_losses.append(train_loss); val_losses.append(val_loss)

        marker = ""
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            marker = "  <- best so far"
        else:
            epochs_no_improve += 1

        if verbose:
            print(f"[{tag}] epoch {epoch:2d} | train MSE {train_loss:.4f} MAE {train_mae:.4f} "
                  f"| val MSE {val_loss:.4f} MAE {val_mae:.4f} | lr {current_lr:.2e}{marker}")

        scheduler.step()

        if epochs_no_improve >= patience:
            if verbose:
                print(f"[{tag}] No val improvement in {patience} epochs — stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    test_loss, test_mae = run_epoch(model, test_loader, criterion, None, device)
    print(f"[{tag}] Final test MSE {test_loss:.4f} | test MAE {test_mae:.4f}")

    torch.save(model.state_dict(), f"{tag}.pt")

    # loss curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train MSE")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="val MSE")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title(f"DLinear training — pred_len {pred_len}")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"loss_curves_{tag}.png", dpi=150); plt.close()
    print(f"saved loss_curves_{tag}.png")

    return test_loss, test_mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="ETTh1.csv")
    parser.add_argument("--pred_len", type=int, default=None,
                         help="if omitted, runs all four: 96, 192, 336, 720")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    args, _ = parser.parse_known_args()

    pred_lens = [args.pred_len] if args.pred_len else [96, 192, 336, 720]
    results = {}
    for pl in pred_lens:
        mse, mae = train_dlinear(pl, csv_path=args.csv_path,
                                  epochs=args.epochs, patience=args.patience)
        results[pl] = (mse, mae)

    if len(results) > 1:
        mses = [v[0] for v in results.values()]
        maes = [v[1] for v in results.values()]
        print("\n===== DLinear Summary (compare to paper's DLinear: avg MSE 0.461, MAE 0.457) =====")
        for pl, (mse, mae) in results.items():
            print(f"pred_len={pl:>4}  MSE={mse:.4f}  MAE={mae:.4f}")
        print(f"\nAverage  MSE={sum(mses)/len(mses):.4f}  MAE={sum(maes)/len(maes):.4f}")
