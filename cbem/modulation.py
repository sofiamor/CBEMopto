"""Modulation index: does simultaneous MC+VC stimulation equal the sum of MC and VC alone?

Compared with the old notebook version (spike-count MI on whole-block rates):
  1. Uses EVOKED responses (spikes after each light onset minus spikes just before
     it), not the mean rate of the whole protocol block. A 5-20 ms pulse once per
     second leaves >97 % of the block as baseline, which used to dominate the MI.
  2. Subtracts the baseline before summing, so MC + VC no longer counts it twice.
  3. Works for inhibited cells too: the old (sim-(mc+vc))/(sim+mc+vc) breaks when
     responses are negative (denominator near zero or negative).
  4. Gives a 95 % bootstrap confidence interval (resampling pulses), so
     "supralinear / sublinear / linear" is a statistical call, not just a sign.
  5. Warns about problems seen in the data: different light durations between the
     single-pulse conditions, and neurons that go silent for a whole protocol block.

Two parts:
  observed_modulation(raw, ...)  from the raw .mat matrix (no model needed)
  model_modulation(params, ...)  what the fitted CBEM predicts for the same pulses
"""
import numpy as np

from .config import FS_ACQ

# protocol ids: (VC, MC, simultaneous)
PAIRS = {'single': (1, 3, 5), 'train': (2, 4, 6)}
# response window after each light onset (s); the same length is used before it
WINDOWS = {'single': 0.030, 'train': 0.100}


# ── helpers ──────────────────────────────────────────────────────────────────
def light_onsets(stim):
    """Indices where the laser goes 0 -> 1 (a block that starts with the laser
    already on counts its first sample as an onset)."""
    s = np.asarray(stim).astype(np.int8)
    return np.flatnonzero(np.diff(np.r_[0, s]) == 1)


def light_duration_ms(stim, fs):
    s = np.asarray(stim).astype(np.int8)
    d = np.diff(np.r_[0, s, 0])
    on, off = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return float(np.median(off - on)) / fs * 1000 if len(on) else np.nan


def evoked_per_pulse(spikes, onsets, win):
    """Per-pulse (post - pre) spike count in a window of `win` samples, plus the
    pre-pulse counts. Pulses too close to the block edges are skipped."""
    ok = [o for o in onsets if o - win >= 0 and o + win <= len(spikes)]
    pre = np.array([spikes[o - win:o].sum() for o in ok], float)
    post = np.array([spikes[o:o + win].sum() for o in ok], float)
    return post - pre, pre


def _boot(x, n, rng):
    if len(x) == 0:
        return np.full(n, np.nan)
    return rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)


def _ci(samples):
    return np.nanpercentile(samples, [2.5, 97.5])


def _classify(lo, hi, responsive):
    if lo > 0:
        return 'supralinear' + ('' if responsive else ' (no single condition significant)')
    if hi < 0:
        return 'sublinear' + ('' if responsive else ' (no single condition significant)')
    if not responsive:
        return 'no response to any condition'
    return 'linear (not different from sum)'


def _summarise(d_mc, d_vc, d_sim, win_s, n_boot, rng):
    """All rates in Hz. d_* are per-pulse post-pre counts."""
    to_hz = 1.0 / win_s
    b_mc, b_vc, b_sim = (_boot(d, n_boot, rng) * to_hz for d in (d_mc, d_vc, d_sim))
    r_mc, r_vc, r_sim = (np.mean(d) * to_hz if len(d) else np.nan
                         for d in (d_mc, d_vc, d_sim))
    diff = b_sim - (b_mc + b_vc)
    norm = np.abs(b_mc) + np.abs(b_vc)
    with np.errstate(divide='ignore', invalid='ignore'):
        ai = np.where(norm > 0, diff / norm, np.nan)
    lo, hi = _ci(diff)
    # a condition "responds" if its bootstrap CI excludes 0
    responsive = any((_ci(b)[0] > 0) or (_ci(b)[1] < 0) for b in (b_mc, b_vc, b_sim))
    ai_val = (r_sim - (r_mc + r_vc)) / (abs(r_mc) + abs(r_vc)) if (abs(r_mc) + abs(r_vc)) > 0 else np.nan
    return {
        'R_mc (Hz)': r_mc, 'R_vc (Hz)': r_vc, 'R_sim (Hz)': r_sim,
        'Linear sum (Hz)': r_mc + r_vc,
        'Sim - sum (Hz)': r_sim - (r_mc + r_vc),
        'Sim - sum CI low': lo, 'Sim - sum CI high': hi,
        'Additivity index': ai_val,
        'AI CI low': _ci(ai)[0], 'AI CI high': _ci(ai)[1],
        'Result': _classify(lo, hi, responsive),
        'n pulses (mc/vc/sim)': f'{len(d_mc)}/{len(d_vc)}/{len(d_sim)}',
    }


# ── observed ─────────────────────────────────────────────────────────────────
def observed_modulation(raw, fs=FS_ACQ, neurons=None, n_boot=2000, seed=0,
                        windows=WINDOWS, include_old_mi=True):
    """Additivity of MC + VC from the raw .mat matrix.

    raw:      the matrix from the .mat file (col 0 protocol, col 1 laser,
              col 2+ spikes), at any sample rate given by `fs`.
    neurons:  list of neuron numbers (1 = first spike column); default all.
    Returns a list of dicts, one per (neuron, single/train), ready for
    pandas.DataFrame(...). Read 'Result' and 'Additivity index':
        AI = (R_sim - (R_mc + R_vc)) / (|R_mc| + |R_vc|)
        AI = 0 linear, > 0 supralinear, < 0 sublinear;
        AI = -1 means simultaneous gives no response at all.
    """
    raw = np.asarray(raw)
    prot = raw[:, 0].astype(int)
    stim = raw[:, 1]
    n_neu = raw.shape[1] - 2
    neurons = neurons or list(range(1, n_neu + 1))
    rng = np.random.default_rng(seed)

    blocks = {p: np.flatnonzero(prot == p) for p in range(1, 7)}
    onsets = {p: light_onsets(stim[idx]) for p, idx in blocks.items()}
    light = {p: light_duration_ms(stim[idx], fs) for p, idx in blocks.items()}

    rows = []
    for n in neurons:
        spk = raw[:, n + 1].astype(np.int32)
        block_rate = {p: spk[idx].sum() / (len(idx) / fs) if len(idx) else np.nan
                      for p, idx in blocks.items()}
        med_rate = np.nanmedian(list(block_rate.values()))
        for kind, (p_vc, p_mc, p_sim) in PAIRS.items():
            win = int(windows[kind] * fs)
            d = {}
            for p in (p_vc, p_mc, p_sim):
                y = spk[blocks[p]]
                d[p], _ = evoked_per_pulse(y, onsets[p], win)
            row = {'Neuron': n, 'Stimulus': kind}
            row.update(_summarise(d[p_mc], d[p_vc], d[p_sim], windows[kind], n_boot, rng))

            warn = []
            durs = [light[p_vc], light[p_mc], light[p_sim]]
            if np.nanmax(durs) - np.nanmin(durs) > 1:
                warn.append(f'light differs: VC {durs[0]:.0f} / MC {durs[1]:.0f} / '
                            f'Sim {durs[2]:.0f} ms, so the sum is not like-for-like')
            for p in (p_vc, p_mc, p_sim):
                if med_rate >= 2 and block_rate[p] < 0.1 * med_rate:
                    warn.append(f'near-silent in whole P{p} block '
                                f'({block_rate[p]:.2f} vs {med_rate:.2f} Hz): unit lost?')
            if any(len(d[p]) < 10 for p in d):
                warn.append('fewer than 10 pulses in a condition: wide CI')
            row['Warnings'] = '; '.join(warn)

            if include_old_mi:   # the original notebook formula, for comparison
                r_mc, r_vc, r_sim = block_rate[p_mc], block_rate[p_vc], block_rate[p_sim]
                den = r_sim + r_mc + r_vc
                row['Old MI (whole-block)'] = (r_sim - (r_mc + r_vc)) / den if den else np.nan
            rows.append(row)
    return rows


# ── model ────────────────────────────────────────────────────────────────────
def model_modulation(params, fl, hl, X, protocol, dt, biophys, make_design_matrix,
                     forward, windows=WINDOWS):
    """What the fitted CBEM predicts for the same comparison, with no spike
    history (stimulus-driven rate only).

    For every simultaneous pulse it runs the model three times on the same
    stretch of stimulus: both lasers, MC column only, VC column only. So the
    three conditions share timing and light duration exactly.

    X:        (T, 2) stimulus at the model rate, col 0 = MC, col 1 = VC.
    protocol: (T,) protocol id at the model rate.
    make_design_matrix, forward: pass your own functions
              (from cbem.data import make_design_matrix; from cbem.model import forward).

    How to read it: the model has no MC x VC interaction term, so any
    non-zero additivity here comes only from the neuron's own nonlinearity
    (softplus, voltage, threshold). Observed minus model = interaction the
    model does not explain. If protocol 6 was in the fit, the filters already
    absorbed part of the interaction: fit on protocols 2 and 4 only for a clean
    comparison.
    """
    X = np.asarray(X, float)
    protocol = np.asarray(protocol).astype(int)
    rows = []
    for kind, (_, _, p_sim) in PAIRS.items():
        win = int(round(windows[kind] / dt))
        idx = np.flatnonzero(protocol == p_sim)
        if len(idx) == 0:
            continue
        on = idx[light_onsets(X[idx, 0])]
        pad = fl  # stimulus history the filters need before the window
        resp = {'sim': [], 'mc': [], 'vc': []}
        for o in on:
            a, b = o - win - pad, o + win
            if a < 0 or b > len(X):
                continue
            Xw = X[a:b]
            for key, Xc in (('sim', Xw),
                            ('mc', np.c_[Xw[:, 0], np.zeros(len(Xw))]),
                            ('vc', np.c_[np.zeros(len(Xw)), Xw[:, 1]])):
                Xd = make_design_matrix(Xc, fl, [(0, len(Xc))])
                lam = forward(params, Xd, None, fl, hl, dt, biophys)['lam'] / dt  # Hz
                pre = lam[pad:pad + win].mean()
                post = lam[pad + win:pad + 2 * win].mean()
                resp[key].append(post - pre)
        r = {k: float(np.mean(v)) if v else np.nan for k, v in resp.items()}
        norm = abs(r['mc']) + abs(r['vc'])
        rows.append({
            'Stimulus': kind, 'Model R_mc (Hz)': r['mc'], 'Model R_vc (Hz)': r['vc'],
            'Model R_sim (Hz)': r['sim'], 'Model linear sum (Hz)': r['mc'] + r['vc'],
            'Model sim - sum (Hz)': r['sim'] - (r['mc'] + r['vc']),
            'Model additivity index': (r['sim'] - (r['mc'] + r['vc'])) / norm if norm else np.nan,
            'n pulses': len(resp['sim']),
        })
    return rows


# ── plot ─────────────────────────────────────────────────────────────────────
def plot_modulation(rows, title=''):
    """One panel per stimulus type: bars for MC, VC, MC+VC and Sim per neuron,
    with the 95 % CI of (Sim - sum) as an error bar on the Sim bar."""
    import matplotlib.pyplot as plt
    kinds = [k for k in PAIRS if any(r['Stimulus'] == k for r in rows)]
    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 4), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        rs = [r for r in rows if r['Stimulus'] == kind]
        x = np.arange(len(rs))
        w = 0.2
        ax.bar(x - 1.5 * w, [r['R_mc (Hz)'] for r in rs], w, label='MC', color='royalblue')
        ax.bar(x - 0.5 * w, [r['R_vc (Hz)'] for r in rs], w, label='VC', color='tomato')
        ax.bar(x + 0.5 * w, [r['Linear sum (Hz)'] for r in rs], w, label='MC + VC',
               color='lightgray', edgecolor='k')
        sim = np.array([r['R_sim (Hz)'] for r in rs])
        lin = np.array([r['Linear sum (Hz)'] for r in rs])
        err = np.array([[s - (l + r['Sim - sum CI low']), (l + r['Sim - sum CI high']) - s]
                        for s, l, r in zip(sim, lin, rs)]).T
        ax.bar(x + 1.5 * w, sim, w, yerr=np.abs(err), capsize=3, ecolor='gray', label='Simultaneous', color='k')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f'n{r["Neuron"]}\nAI={r["Additivity index"]:+.2f}' for r in rs])
        ax.set_ylabel('Evoked response (Hz, post - pre)')
        ax.set_title(f'{title} {kind} pulses'.strip())
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
