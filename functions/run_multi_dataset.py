import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from timemixer import (
    Configs, Model, run_epoch, plot_loss_curves, plot_forecasts,
    plot_multiscale_predictions, plot_scale_predictions,
    ETTh1Dataset as _WindowDataset, load_etth1,
)
from baseline_dlinear import DLinear

torch.manual_seed(2000)
np.random.seed(2000)


def load_ratio_split_dataset(path: str, seq_len: int, pred_len: int,
                              train_ratio: float = 0.7, test_ratio: float = 0.2):

    df = pd.read_csv(path)
    date_col = df.columns[0]
    cols = [c for c in df.columns if c != date_col]
    data = df[cols].values.astype(np.float32)
    n = len(data)

    num_train = int(n * train_ratio)
    num_test = int(n * test_ratio)
    num_val = n - num_train - num_test

    train_raw = data[:num_train]
    val_raw = data[num_train - seq_len: num_train + num_val]
    test_raw = data[num_train + num_val - seq_len:]

    mean, std = train_raw.mean(axis=0), train_raw.std(axis=0)
    std[std == 0] = 1.0
    norm = lambda x: (x - mean) / std

    train_ds = _WindowDataset(norm(train_raw), seq_len, pred_len)
    val_ds = _WindowDataset(norm(val_raw), seq_len, pred_len)
    test_ds = _WindowDataset(norm(test_raw), seq_len, pred_len)

    assert len(train_ds) > 0 and len(val_ds) > 0 and len(test_ds) > 0, \
        "empty split — check seq_len/pred_len vs data length"

    return train_ds, val_ds, test_ds, (mean, std), len(cols), cols


DATASET_LOADERS = {
    "etth1": lambda path, seq_len, pred_len: load_etth1(path, seq_len, pred_len),
    "weather": lambda path, seq_len, pred_len: load_ratio_split_dataset(path, seq_len, pred_len),
    "electricity": lambda path, seq_len, pred_len: load_ratio_split_dataset(path, seq_len, pred_len),
}

# representative channel to plot for each dataset (paper convention: last
# column, typically the main target variate in these benchmark CSVs)
DEFAULT_PLOT_VAR = {
    "etth1": "OT",
    "weather": None,   # resolved to last column at runtime
    "electricity": None,
}


def train_timemixer(dataset, pred_len, csv_path, seq_len=96,
                     M=3, L=2, d_model=16, batch_size=32,
                     epochs=20, lr=1e-2, patience=10,
                     channel_independence=1, verbose=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, test_ds, (mean, std), c_in, cols = DATASET_LOADERS[dataset](
        csv_path, seq_len, pred_len
    )
    var_name = DEFAULT_PLOT_VAR[dataset] or cols[-1]
    tag = f"timemixer_{dataset}_pred{pred_len}"
    if verbose:
        print(f"[{tag}] train/val/test windows: "
              f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)} | variates: {c_in} | device: {device}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    configs = Configs(
        seq_len=seq_len, pred_len=pred_len, enc_in=c_in, c_out=c_in,
        down_sampling_layers=M, e_layers=L, d_model=d_model,
        use_norm=1, channel_independence=channel_independence,
    )
    model = Model(configs).to(device)

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
    plot_loss_curves(train_losses, val_losses, save_path=f"loss_curves_{tag}.png")
    plot_forecasts(model, test_ds, mean, std, device, col_names=cols,
                    var_name=var_name, save_path=f"forecasts_{tag}.png")
    plot_multiscale_predictions(model, test_ds, mean, std, device, col_names=cols,
                                 var_name=var_name, num_scales=M,
                                 save_path=f"multiscale_{tag}.png")
    # Figure 4 reproduction: multiscale mixing (final) vs. each individual
    # scale's predictor output, before summing.
    plot_scale_predictions(model, test_ds, mean, std, device, col_names=cols,
                            var_name=var_name, save_path=f"scalepred_{tag}.png")

    return test_loss, test_mae


def train_dlinear_generic(dataset, pred_len, csv_path, seq_len=96,
                           batch_size=32, epochs=20, lr=1e-2, patience=10, verbose=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, test_ds, (mean, std), c_in, cols = DATASET_LOADERS[dataset](
        csv_path, seq_len, pred_len
    )
    tag = f"dlinear_{dataset}_pred{pred_len}"
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

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_mae = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = run_epoch(model, val_loader, criterion, None, device)

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

    return test_loss, test_mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                         choices=["etth1", "weather", "electricity"])
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--pred_len", type=int, default=None,
                         help="if omitted, runs all four: 96, 192, 336, 720")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", type=str, default="timemixer",
                         choices=["timemixer", "dlinear", "both"])
    args, _ = parser.parse_known_args()

    pred_lens = [args.pred_len] if args.pred_len else [96, 192, 336, 720]
    tm_results, dl_results = {}, {}

    for pl in pred_lens:
        if args.model in ("timemixer", "both"):
            mse, mae = train_timemixer(args.dataset, pl, args.csv_path,
                                        epochs=args.epochs, patience=args.patience,
                                        batch_size=args.batch_size)
            tm_results[pl] = (mse, mae)
        if args.model in ("dlinear", "both"):
            mse, mae = train_dlinear_generic(args.dataset, pl, args.csv_path,
                                              epochs=args.epochs, patience=args.patience,
                                              batch_size=args.batch_size)
            dl_results[pl] = (mse, mae)

    def summarize(name, results):
        if len(results) > 1:
            mses = [v[0] for v in results.values()]
            maes = [v[1] for v in results.values()]
            print(f"\n===== {name} on {args.dataset} — Summary =====")
            for pl, (mse, mae) in results.items():
                print(f"pred_len={pl:>4}  MSE={mse:.4f}  MAE={mae:.4f}")
            print(f"Average  MSE={sum(mses)/len(mses):.4f}  MAE={sum(maes)/len(maes):.4f}")

    summarize("TimeMixer", tm_results)
    summarize("DLinear", dl_results)

            print(f"Average  MSE={sum(mses)/len(mses):.4f}  MAE={sum(maes)/len(maes):.4f}")

    summarize("TimeMixer", tm_results)
    summarize("DLinear", dl_results)
