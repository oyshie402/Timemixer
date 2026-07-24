


import torch
import torch.nn as nn


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
    """Sec 3.1: X0 = Embed(X)."""
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(c_in, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, scales: list):
        return [self.dropout(self.proj(x_m)) for x_m in scales]


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

    def forward(self, x_list: list):
        preds = [predictor(x_m) for predictor, x_m in zip(self.predictors, x_list)]
        return torch.stack(preds, dim=0).sum(dim=0)


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

    def forward(self, x_list: list):
        preds = torch.stack(
            [predictor(x_m) for predictor, x_m in zip(self.predictors, x_list)],
            dim=0
        )
        weights = torch.softmax(self.scale_weights, dim=0).view(-1, 1, 1, 1)
        return (preds * weights).sum(dim=0)


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

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
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

        dec_out = self.fmm(XL)  # [B*N, pred_len, 1] if channel-independent, else [B, pred_len, C]

        if self.channel_independence:
            dec_out = dec_out.reshape(B, N, self.pred_len, 1).squeeze(-1).permute(0, 2, 1)  # [B, pred_len, N]

        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        raise ValueError(
            f"This replication only implements forecasting; got task_name={self.task_name!r}"
        )


if __name__ == "__main__":
    # Minimal smoke test: build the model and run a forward pass on random data.
    configs = Configs(seq_len=96, pred_len=96, enc_in=7)
    model = Model(configs)
    x = torch.randn(4, configs.seq_len, configs.enc_in)
    y = model(x)
    print(f"input shape:  {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)}")