"""Loading, 1 kHz downsampling, MC/VC stimulus split and design matrices."""
from dataclasses import dataclass

import numpy as np
import scipy.io as sio

from .config import DOWNSAMPLE, DT, PROTOCOL_NAMES


@dataclass
class Dataset:
    protocol_1k: np.ndarray   # (T,) protocol id per 1 kHz bin
    stim_1k: np.ndarray       # (T,) merged laser channel
    spk_1k: dict              # {1: (T,), 2: (T,), ...} spike trains
    X_full: np.ndarray        # (T, 2): col 0 = MC, col 1 = VC
    n_neurons: int

    @property
    def T(self):
        return len(self.stim_1k)


@dataclass
class FitSet:
    X: np.ndarray             # (T_fit, 2) stimulus of the fitted protocols
    protocol: np.ndarray      # (T_fit,)
    spikes: dict              # {'neuron1': (T_fit,), ...}
    segments: list            # [(start, stop), ...] contiguous same-protocol runs

    @property
    def T(self):
        return len(self.X)


# ── loading ──────────────────────────────────────────────────────────────────
def load_raw(mat_file, variable=None):
    """Load the matrix from a .mat file. If `variable` is None the file must
    contain exactly one variable."""
    mat = sio.loadmat(mat_file)
    keys = [k for k in mat if not k.startswith('__')]
    if variable is None:
        if len(keys) != 1:
            raise ValueError(f'Several variables in file, pick one: {keys}')
        variable = keys[0]
    if variable not in mat:
        raise KeyError(f'{variable!r} not found. Variables: {keys}')
    return mat[variable]


def _downsample_max(vec, factor=DOWNSAMPLE):
    """Max over each window: any window containing a 1 becomes 1."""
    n = (len(vec) // factor) * factor
    return vec[:n].reshape(-1, factor).max(axis=1).astype(np.float32)


def _downsample_first(vec, factor=DOWNSAMPLE):
    """First sample of each window (protocol id is constant inside a window)."""
    n = (len(vec) // factor) * factor
    return vec[:n].reshape(-1, factor)[:, 0].astype(np.int8)


def prepare_dataset(raw):
    """raw columns: A = protocol id, B = merged laser, C.. = spike trains."""
    n_neurons = raw.shape[1] - 2
    protocol_1k = _downsample_first(raw[:, 0].astype(int))
    stim_1k = _downsample_max(raw[:, 1].astype(float))
    spk_1k = {i: _downsample_max(raw[:, i + 1].astype(float)).astype(np.int8)
              for i in range(1, n_neurons + 1)}

    # Protocols 1,2 -> VC only | 3,4 -> MC only | 5,6 -> both channels
    X_mc = np.zeros(len(stim_1k), dtype=np.float32)
    X_vc = np.zeros(len(stim_1k), dtype=np.float32)
    vc_only = np.isin(protocol_1k, [1, 2])
    mc_only = np.isin(protocol_1k, [3, 4])
    both = np.isin(protocol_1k, [5, 6])
    X_vc[vc_only | both] = stim_1k[vc_only | both]
    X_mc[mc_only | both] = stim_1k[mc_only | both]

    return Dataset(protocol_1k, stim_1k, spk_1k,
                   np.stack([X_mc, X_vc], axis=1), n_neurons)


def describe_dataset(ds):
    print(f'{ds.T} bins = {ds.T * DT:.1f} s at 1 kHz, {ds.n_neurons} neurons')
    head = f'{"Proto":<6}{"Name":<30}{"Dur(s)":>9}{"MC_ON":>8}{"VC_ON":>8}'
    head += ''.join(f'{"spk" + str(i):>7}' for i in range(1, ds.n_neurons + 1))
    print(head)
    for pid in range(1, 7):
        m = ds.protocol_1k == pid
        if not m.any():
            continue
        row = (f'{pid:<6}{PROTOCOL_NAMES[pid]:<30}{m.sum() * DT:>9.1f}'
               f'{int(ds.X_full[m, 0].sum()):>8}{int(ds.X_full[m, 1].sum()):>8}')
        row += ''.join(f'{int(ds.spk_1k[i][m].sum()):>7}'
                       for i in range(1, ds.n_neurons + 1))
        print(row)


# ── segments ─────────────────────────────────────────────────────────────────
def runs(labels):
    """Contiguous runs of equal value -> [(start, stop), ...]."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, change]
    stops = np.r_[change, len(labels)]
    return list(zip(starts.tolist(), stops.tolist()))


def _offsets(pieces):
    """[(a, b), ...] source slices -> [(0, n0), (n0, n0+n1), ...] in a new array."""
    out, cum = [], 0
    for a, b in pieces:
        out.append((cum, cum + b - a))
        cum += b - a
    return out


def build_fit_set(ds, fit_protocols):
    """Concatenate the runs of the fitted protocols. Each run stays a separate
    segment so the model never integrates across a jump between runs."""
    pieces = [(a, b) for a, b in runs(ds.protocol_1k)
              if ds.protocol_1k[a] in fit_protocols]
    if not pieces:
        raise ValueError(f'No data for protocols {fit_protocols}')
    idx = np.concatenate([np.arange(a, b) for a, b in pieces])
    return FitSet(
        X=ds.X_full[idx],
        protocol=ds.protocol_1k[idx],
        spikes={f'neuron{i}': ds.spk_1k[i][idx] for i in range(1, ds.n_neurons + 1)},
        segments=_offsets(pieces),
    )


def balanced_subset(fit, total_bins):
    """Take the first ~total_bins/n_protocols bins of every fitted protocol,
    so a short search subset is not dominated by one protocol."""
    protos = np.unique(fit.protocol)
    per = total_bins // len(protos)
    pieces = []
    for p in protos:
        used = 0
        for a, b in fit.segments:
            if fit.protocol[a] != p:
                continue
            take = min(b - a, per - used)
            if take <= 0:
                break
            pieces.append((a, a + take))
            used += take
    idx = np.concatenate([np.arange(a, b) for a, b in pieces])
    return FitSet(
        X=fit.X[idx],
        protocol=fit.protocol[idx],
        spikes={k: v[idx] for k, v in fit.spikes.items()},
        segments=_offsets(pieces),
    )


def train_val_segments(segments, val_frac):
    """Hold out the last val_frac of EVERY segment, so validation covers all
    protocols (a single tail split would be mostly the last protocol).
    Returns two lists of contiguous (start, stop) slices."""
    train, val = [], []
    for a, b in segments:
        n_val = int((b - a) * val_frac)
        if n_val == 0 or b - a - n_val == 0:
            train.append((a, b))
            continue
        train.append((a, b - n_val))
        val.append((b - n_val, b))
    return train, val


# ── design matrix ────────────────────────────────────────────────────────────
def make_design_matrix(X, filter_len, segments=None):
    """(T, 2*filter_len) lagged stimulus matrix. Columns 0..fl-1 are MC lags
    0..fl-1, columns fl..2fl-1 are VC lags. Lags never cross a segment start."""
    T = len(X)
    if segments is None:
        segments = [(0, T)]
    Xd = np.zeros((T, 2 * filter_len), dtype=np.float32)
    for a, b in segments:
        n = b - a
        for ch in range(2):
            xpad = np.concatenate([np.zeros(filter_len - 1, dtype=np.float32),
                                   X[a:b, ch].astype(np.float32)])
            c0 = ch * filter_len
            for lag in range(filter_len):
                Xd[a:b, c0 + lag] = xpad[filter_len - 1 - lag: n + filter_len - 1 - lag]
    return Xd
