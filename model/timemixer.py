import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(2000)
np.random.seed(2000)


# ========================================================================
# CONFIG
# ========================================================================
class Configs:
    def __init__(self, seq_len, pred_len, enc_in, c_out=None,
                 down_sampling_window=2, down_sampling_layers=3,
                 e_layers=2, d_model=16, d_ff=None, moving_avg=25,
                 dropout=0.1, use_norm=1, reverse_mixing=False,
                 weighted_fmm=False, task_name='long_term_forecast',
                 channel_independence=1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out or enc_in
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers   # M in the paper
        self.e_layers = e_layers                            # L in the paper
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.moving_avg = moving_avg
        self.dropout = dropout
        self.use_norm = use_norm
        self.reverse_mixing = reverse_mixing
        self.weighted_fmm = weighted_fmm
        self.task_name = task_name
        self.channel_independence = channel_independence


# ========================================================================
# PER-SCALE NORMALIZATION (RevIN-style)
# ========================================================================
class Normalize(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, non_norm=False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.non_norm = non_norm
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str):
        if mode == 'norm':
            if self.non_norm:
                return x
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.non_norm:
                return x
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps ** 2)
            return x * self.stdev + self.mean
        raise ValueError(f"mode must be 'norm' or 'denorm', got {mode!r}")


# ========================================================================
# MODEL BUILDING BLOCKS — Eq. 1-6
# ========================================================================
class MultiscaleDownsampling(nn.Module):
    """Sec 3.1: X = {x0,...,xM} via repeated average pooling."""
    def __init__(self, M: int, window: int = 2):
        super().__init__()
        self.M = M
        self.pool = nn.AvgPool1d(kernel_size=window, stride=window)

    def forward(self, x: torch.Tensor):
        scales = [x]
        cur = x.permute(0, 2, 1)
        for _ in range(self.M):
            cur = self.pool(cur)
            scales.append(cur.permute(0, 2, 1))
        return scales


class Embed(nn.Module):
    """
    Sec 3.1: X0 = Embed(X).

    Matches the official repo's TokenEmbedding (used inside
    DataEmbedding_wo_pos): a 1D convolution with kernel_size=3 and
    circular padding, instead of a plain per-timestep nn.Linear. This
    lets each embedded timestep borrow local context from its immediate
    neighbors, which matters most at coarse scales where very few
    timesteps are available (e.g. scale 3 with seq_len=96, M=3 sees only
    12 points) — a plain per-timestep linear has no way to use
    neighboring information there, which was found to make coarse-scale
    predictors collapse to a near-constant output.
    """
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        padding = 1  # kernel_size=3, padding=1 preserves sequence length
        self.token_conv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model, kernel_size=3,
            padding=padding, padding_mode='circular', bias=False
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.dropout = nn.Dropout(dropout)

    def forward(self, scales: list):
        # x_m: [B, len_m, c_in] -> conv expects [B, c_in, len_m] -> back to [B, len_m, d_model]
        return [
            self.dropout(self.token_conv(x_m.permute(0, 2, 1)).transpose(1, 2))
            for x_m in scales
        ]


class SeriesDecomp(nn.Module):
    """Sec 3.2: Autoformer-style moving-average seasonal/trend split."""
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        assert kernel_size % 2 == 1, "use an odd kernel_size for exact length preservation"
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        padded = torch.cat([front, x, end], dim=1)
        trend = self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class TemporalMixingLayer(nn.Module):
    """Two linear layers + GELU along the temporal dimension (Eq. 4 & 5)."""
    def __init__(self, in_len: int, out_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_len, out_len),
            nn.GELU(),
            nn.Linear(out_len, out_len),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)


class MultiScaleSeasonMixing(nn.Module):
    """Eq. 4 — Bottom-Up-Mixing: fine -> coarse."""
    def __init__(self, scale_lengths: list):
        super().__init__()
        self.layers = nn.ModuleList([
            TemporalMixingLayer(scale_lengths[m - 1], scale_lengths[m])
            for m in range(1, len(scale_lengths))
        ])

    def forward(self, seasonal_list: list):
        out = [seasonal_list[0]]
        for m in range(1, len(seasonal_list)):
            out.append(seasonal_list[m] + self.layers[m - 1](out[m - 1]))
        return out


class MultiScaleTrendMixing(nn.Module):
    """Eq. 5 — Top-Down-Mixing: coarse -> fine."""
    def __init__(self, scale_lengths: list):
        super().__init__()
        M = len(scale_lengths) - 1
        self.layers = nn.ModuleList([
            TemporalMixingLayer(scale_lengths[m + 1], scale_lengths[m])
            for m in range(M)
        ])

    def forward(self, trend_list: list):
        M = len(trend_list) - 1
        out = [None] * (M + 1)
        out[M] = trend_list[M]
        for m in range(M - 1, -1, -1):
            out[m] = trend_list[m] + self.layers[m](out[m + 1])
        return out


class PastDecomposableMixing(nn.Module):
    """Eq. 3 — a single PDM layer: decompose, mix, residual FeedForward."""
    def __init__(self, scale_lengths: list, d_model: int, d_ff: int = None,
                 decomp_kernel: int = 25, dropout: float = 0.1, reverse_mixing: bool = False):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.decomp = SeriesDecomp(decomp_kernel)
        if reverse_mixing:
            # Ablation (paper Table 5, case 7): seasonal gets top-down,
            # trend gets bottom-up — opposite of the official design.
            self.seasonal_mix = MultiScaleTrendMixing(scale_lengths)
            self.trend_mix = MultiScaleSeasonMixing(scale_lengths)
        else:
            self.seasonal_mix = MultiScaleSeasonMixing(scale_lengths)
            self.trend_mix = MultiScaleTrendMixing(scale_lengths)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x_list: list):
        seasonal_list, trend_list = [], []
        for x_m in x_list:
            s_m, t_m = self.decomp(x_m)
            seasonal_list.append(s_m)
            trend_list.append(t_m)
        s_mixed = self.seasonal_mix(seasonal_list)
        t_mixed = self.trend_mix(trend_list)
        return [
            x_m + self.feed_forward(s + t)
            for x_m, s, t in zip(x_list, s_mixed, t_mixed)
        ]


class ScalePredictor(nn.Module):
    """Predictor_m: one linear layer for length-F regression + channel projection."""
    def __init__(self, in_len: int, pred_len: int, d_model: int, c_out: int):
        super().__init__()
        self.temporal_proj = nn.Linear(in_len, pred_len)
        self.channel_proj = nn.Linear(d_model, c_out)

    def forward(self, x_m: torch.Tensor):
        x = self.temporal_proj(x_m.permute(0, 2, 1)).permute(0, 2, 1)
        return self.channel_proj(x)


class FutureMultipredictorMixing(nn.Module):
    """Eq. 6 — ensemble of M+1 per-scale predictors, summed."""
    def __init__(self, scale_lengths: list, pred_len: int, d_model: int, c_out: int):
        super().__init__()
        self.predictors = nn.ModuleList([
            ScalePredictor(length, pred_len, d_model, c_out)
            for length in scale_lengths
        ])

    def forward(self, x_list: list, return_per_scale: bool = False):
        preds = [predictor(x_m) for predictor, x_m in zip(self.predictors, x_list)]
        summed = torch.stack(preds, dim=0).sum(dim=0)
        if return_per_scale:
            return summed, preds  # preds: list of x_hat_m (Eq. 6, before summing)
        return summed


class WeightedFutureMultipredictorMixing(nn.Module):
    """Original extension (not in the paper): learns one softmax'd weight
    per scale instead of an equal-weight sum."""
    def __init__(self, scale_lengths: list, pred_len: int, d_model: int, c_out: int):
        super().__init__()
        self.predictors = nn.ModuleList([
            ScalePredictor(length, pred_len, d_model, c_out)
            for length in scale_lengths
        ])
        self.scale_weights = nn.Parameter(torch.ones(len(scale_lengths)))

    def forward(self, x_list: list, return_per_scale: bool = False):
        raw_preds = [predictor(x_m) for predictor, x_m in zip(self.predictors, x_list)]
        preds = torch.stack(raw_preds, dim=0)
        weights = torch.softmax(self.scale_weights, dim=0).view(-1, 1, 1, 1)
        summed = (preds * weights).sum(dim=0)
        if return_per_scale:
            weighted_preds = [(preds[i] * weights[i]) for i in range(len(raw_preds))]
            return summed, weighted_preds
        return summed


# ========================================================================
# MODEL — matches official repo's interface; supports channel-independence
# ========================================================================
class Model(nn.Module):
    def __init__(self, configs: Configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.channel_independence = configs.channel_independence

        self.downsample = MultiscaleDownsampling(configs.down_sampling_layers, configs.down_sampling_window)

        # channel-independence: embed a single scalar channel (c_in=1)
        # instead of all variates jointly — matches official repo's
        # default for ETT-family datasets. This effectively multiplies
        # the batch size by the number of variates during PDM/FMM, which
        # regularizes training much more than joint embedding.
        embed_c_in = 1 if self.channel_independence else configs.enc_in
        self.embed = Embed(embed_c_in, configs.d_model, configs.dropout)

        with torch.no_grad():
            dummy_scales = self.downsample(torch.zeros(1, configs.seq_len, configs.enc_in))
            self.scale_lengths = [s.shape[1] for s in dummy_scales]

        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                self.scale_lengths, configs.d_model, configs.d_ff,
                configs.moving_avg, configs.dropout, configs.reverse_mixing,
            )
            for _ in range(configs.e_layers)
        ])

        fmm_c_out = 1 if self.channel_independence else configs.c_out
        fmm_cls = WeightedFutureMultipredictorMixing if configs.weighted_fmm else FutureMultipredictorMixing
        self.fmm = fmm_cls(self.scale_lengths, configs.pred_len, configs.d_model, fmm_c_out)

        self.normalize_layers = nn.ModuleList([
            Normalize(configs.enc_in, affine=True, non_norm=(configs.use_norm == 0))
            for _ in range(configs.down_sampling_layers + 1)
        ])

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, return_per_scale=False):
        B, T, N = x_enc.shape

        X = self.downsample(x_enc)
        X = [self.normalize_layers[i](x_m, 'norm') for i, x_m in enumerate(X)]

        if self.channel_independence:
            # [B, len_m, N] -> [B*N, len_m, 1]
            X = [x_m.permute(0, 2, 1).contiguous().reshape(B * N, x_m.shape[1], 1) for x_m in X]

        X0 = self.embed(X)

        XL = X0
        for block in self.pdm_blocks:
            XL = block(XL)

        if not return_per_scale:
            dec_out = self.fmm(XL)  # [B*N, pred_len, 1] if channel-independent, else [B, pred_len, C]
            if self.channel_independence:
                dec_out = dec_out.reshape(B, N, self.pred_len, 1).squeeze(-1).permute(0, 2, 1)  # [B, pred_len, N]
            dec_out = self.normalize_layers[0](dec_out, 'denorm')
            return dec_out

        # per-scale mode — reproduces paper Figure 4 (x_hat_m in Eq. 6, before summing)
        dec_out, per_scale_preds = self.fmm(XL, return_per_scale=True)

        def _reshape_and_denorm(t):
            if self.channel_independence:
                t = t.reshape(B, N, self.pred_len, 1).squeeze(-1).permute(0, 2, 1)  # [B, pred_len, N]
            return self.normalize_layers[0](t, 'denorm')

        dec_out = _reshape_and_denorm(dec_out)
        per_scale_preds = [_reshape_and_denorm(p) for p in per_scale_preds]
        return dec_out, per_scale_preds

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        raise ValueError(
            f"This replication only implements forecasting; got task_name={self.task_name!r}"
        )


# ========================================================================
# DATA
# ========================================================================
class ETTh1Dataset(Dataset):
    """Generic sliding-window dataset. Named for its original use with
    ETTh1, but used as the shared window Dataset for all datasets in
    run_multi_dataset.py too."""
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def load_etth1(path: str, seq_len: int, pred_len: int):
    """ETT-specific loader: fixed 12/4/4-month train/val/test split, as
    used across the Informer/Autoformer/TimeMixer line of papers."""
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c != "date"]
    data = df[cols].values.astype(np.float32)
    n = len(data)
    train_end = 12 * 30 * 24
    val_end = train_end + 4 * 30 * 24
    test_end = min(val_end + 4 * 30 * 24, n)
    train_raw = data[:train_end]
    val_raw = data[train_end - seq_len: val_end]
    test_raw = data[val_end - seq_len: test_end]
    mean, std = train_raw.mean(axis=0), train_raw.std(axis=0)
    std[std == 0] = 1.0
    norm = lambda x: (x - mean) / std
    train_ds = ETTh1Dataset(norm(train_raw), seq_len, pred_len)
    val_ds = ETTh1Dataset(norm(val_raw), seq_len, pred_len)
    test_ds = ETTh1Dataset(norm(test_raw), seq_len, pred_len)
    assert len(train_ds) > 0 and len(val_ds) > 0 and len(test_ds) > 0, \
        "empty split — check seq_len/pred_len vs data length"
    return train_ds, val_ds, test_ds, (mean, std), len(cols), cols


# ========================================================================
# TRAIN / EVAL HELPERS (shared by run_multi_dataset.py and baseline_dlinear.py)
# ========================================================================
def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_mae, n_batches = 0.0, 0.0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(pred - y)).item()
            n_batches += 1
    return total_loss / n_batches, total_mae / n_batches


def plot_loss_curves(train_losses, val_losses, save_path="loss_curves.png"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_losses, label="train MSE")
    plt.plot(epochs, val_losses, label="val MSE")
    best_ep = int(np.argmin(val_losses)) + 1
    plt.axvline(best_ep, color="gray", linestyle="--", alpha=0.6, label=f"best val (epoch {best_ep})")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("Training / validation loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"saved {save_path}")


@torch.no_grad()
def plot_forecasts(model, test_ds, mean, std, device, col_names,
                    var_name="OT", n_samples=3, save_path="forecasts.png"):
    model.eval()
    var_idx = col_names.index(var_name)
    v_mean, v_std = mean[var_idx], std[var_idx]
    idxs = np.linspace(0, len(test_ds) - 1, n_samples, dtype=int)
    fig, axes = plt.subplots(n_samples, 1, figsize=(8, 3 * n_samples), sharex=True)
    if n_samples == 1:
        axes = [axes]
    for ax, i in zip(axes, idxs):
        x, y = test_ds[i]
        pred = model(x.unsqueeze(0).to(device)).cpu().squeeze(0)
        y_true = y[:, var_idx].numpy() * v_std + v_mean
        y_pred = pred[:, var_idx].numpy() * v_std + v_mean
        x_hist = x[:, var_idx].numpy() * v_std + v_mean
        hist_t = np.arange(-len(x_hist), 0)
        fut_t = np.arange(0, len(y_true))
        ax.plot(hist_t, x_hist, color="gray", label="history")
        ax.plot(fut_t, y_true, color="black", label="ground truth")
        ax.plot(fut_t, y_pred, color="tab:red", linestyle="--", label="forecast")
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_title(f"test window {i}"); ax.legend(fontsize=8)
    axes[-1].set_xlabel("time step (0 = forecast start)")
    fig.suptitle(f"TimeMixer forecast vs. ground truth — variate '{var_name}'")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"saved {save_path}")


@torch.no_grad()
def get_multiscale_series(series_1d: np.ndarray, num_scales: int, decomp_kernel: int = 25):
    x = torch.from_numpy(series_1d).float().view(1, -1, 1)
    decomp = SeriesDecomp(decomp_kernel)
    mixing_list, season_list, trend_list = [], [], []
    cur = x
    for m in range(num_scales + 1):
        s, t = decomp(cur)
        mixing_list.append(cur.squeeze().numpy())
        season_list.append(s.squeeze().numpy())
        trend_list.append(t.squeeze().numpy())
        if m < num_scales:
            cur = torch.nn.functional.avg_pool1d(
                cur.permute(0, 2, 1), kernel_size=2, stride=2
            ).permute(0, 2, 1)
    return mixing_list, season_list, trend_list


@torch.no_grad()
def plot_multiscale_predictions(model, test_ds, mean, std, device, col_names,
                                 var_name="OT", sample_idx=0, num_scales=3,
                                 decomp_kernel=25, save_path="multiscale_predictions.png"):
    model.eval()
    var_idx = col_names.index(var_name)
    v_mean, v_std = mean[var_idx], std[var_idx]

    x, y = test_ds[sample_idx]
    pred = model(x.unsqueeze(0).to(device)).cpu().squeeze(0)

    x_hist = x[:, var_idx].numpy() * v_std + v_mean
    y_true = y[:, var_idx].numpy() * v_std + v_mean
    y_pred = pred[:, var_idx].numpy() * v_std + v_mean

    full_true = np.concatenate([x_hist, y_true])
    full_pred = np.concatenate([x_hist, y_pred])

    mix_t, season_t, trend_t = get_multiscale_series(full_true, num_scales, decomp_kernel)
    mix_p, season_p, trend_p = get_multiscale_series(full_pred, num_scales, decomp_kernel)

    scale_ids = [0, num_scales]
    row_labels = ["Mixing", "Season", "Trend"]
    row_data_t = [mix_t, season_t, trend_t]
    row_data_p = [mix_p, season_p, trend_p]

    fig, axes = plt.subplots(3, len(scale_ids), figsize=(6 * len(scale_ids), 9))
    for r in range(3):
        for c, scale in enumerate(scale_ids):
            ax = axes[r, c]
            ax.plot(row_data_t[r][scale], label="GroundTruth", color="tab:blue")
            ax.plot(row_data_p[r][scale], label="Prediction", color="tab:orange")
            if r == 0:
                ax.set_title(f"Scale{scale}")
            if c == 0:
                ax.set_ylabel(row_labels[r])
            ax.legend(fontsize=7)

    fig.suptitle(f"(c) Multiscale Season-trend Predictions\n"
                 f"(input-{len(x_hist)}-predict-{len(y_true)})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"saved {save_path}")


@torch.no_grad()
def plot_scale_predictions(model, test_ds, mean, std, device, col_names,
                            var_name="OT", sample_idx=0, save_path="scale_predictions.png"):
    """
    Reproduces paper Figure 4: ground truth vs. prediction for (a) the
    final ensembled ("Multiscale mixing") output and (b)-(e) each
    individual scale's predictor output (x_hat_m in Eq. 6, before
    summing), plotted side by side over the prediction horizon only.
    """
    model.eval()
    var_idx = col_names.index(var_name)
    v_mean, v_std = mean[var_idx], std[var_idx]

    x, y = test_ds[sample_idx]
    x_batch = x.unsqueeze(0).to(device)

    dec_out, per_scale_preds = model.forecast(x_batch, return_per_scale=True)
    dec_out = dec_out.cpu().squeeze(0)
    per_scale_preds = [p.cpu().squeeze(0) for p in per_scale_preds]

    y_true = y[:, var_idx].numpy() * v_std + v_mean
    y_mixing = dec_out[:, var_idx].numpy() * v_std + v_mean
    y_scales = [p[:, var_idx].numpy() * v_std + v_mean for p in per_scale_preds]

    num_scales = len(y_scales)
    titles = ["(a) Multiscale mixing"] + [f"({chr(ord('b') + m)}) Scale {m}" for m in range(num_scales)]
    curves = [y_mixing] + y_scales

    fig, axes = plt.subplots(1, len(curves), figsize=(4 * len(curves), 3.2), sharey=True)
    for ax, title, curve in zip(axes, titles, curves):
        ax.plot(y_true, label="GroundTruth", color="tab:blue")
        ax.plot(curve, label="Prediction", color="tab:orange")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time step")
    axes[0].set_ylabel(f"predict-{len(y_true)}")
    axes[0].legend(fontsize=7, loc="upper right")

    fig.suptitle(f"Predictions from different scales — variate '{var_name}' "
                 f"(input-{len(x)}-predict-{len(y_true)})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"saved {save_path}")
