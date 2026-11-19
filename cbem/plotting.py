"""All figures. Every function takes outdir=None; pass a folder to also save a PDF."""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import DT, FS_OUT, PROTOCOL_NAMES
from .data import runs

BLUE, RED, GREY = 'royalblue', 'tomato', 'gray'


def _finish(fig, outdir, name):
    fig.tight_layout()
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, name), bbox_inches='tight')
    plt.show()


# ── data checks ──────────────────────────────────────────────────────────────
def plot_sanity(ds, protocol, nidx=1, n_bins=10000, outdir=None):
    """Spikes should follow laser pulses by a few ms."""
    idx = np.where(ds.protocol_1k == protocol)[0][:n_bins]
    t = np.arange(len(idx)) * DT
    fig, ax = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
    ax[0].plot(t, ds.X_full[idx, 0], color=BLUE, lw=1); ax[0].set_ylabel('MC laser')
    ax[0].set_title(f'Protocol {protocol}: {PROTOCOL_NAMES.get(protocol, "")}')
    ax[1].plot(t, ds.X_full[idx, 1], color=RED, lw=1); ax[1].set_ylabel('VC laser')
    ax[2].vlines(t[ds.spk_1k[nidx][idx] == 1], 0, 1, color='k', lw=0.5)
    ax[2].set_ylabel(f'neuron{nidx} spikes'); ax[2].set_xlabel('Time (s)')
    _finish(fig, outdir, 'cbem_sanity.pdf')


# ── hyper-parameter search / CV ──────────────────────────────────────────────
def plot_gridsearch(records, grid, best, outdir=None):
    lams, fls, hls = grid['lambda_l2'], grid['filter_len'], grid['hist_len']
    vals = [r['worst_val_per_spk'] for r in records]
    vmin, vmax = min(vals), max(vals)
    fig, axes = plt.subplots(1, len(lams), figsize=(5 * len(lams), 4), squeeze=False)
    for ax, lam in zip(axes[0], lams):
        m = np.full((len(fls), len(hls)), np.nan)
        for r in records:
            if r['lambda'] == lam:
                m[fls.index(r['fl']), hls.index(r['hl'])] = r['worst_val_per_spk']
        im = ax.imshow(m, aspect='auto', cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
        for i in range(len(fls)):
            for j in range(len(hls)):
                if not np.isnan(m[i, j]):
                    ax.text(j, i, f'{m[i, j]:.3f}', ha='center', va='center', fontsize=8)
        if best['lambda'] == lam:
            ax.add_patch(plt.Rectangle((hls.index(best['hl']) - .5, fls.index(best['fl']) - .5),
                                       1, 1, fill=False, edgecolor='k', lw=3))
        ax.set_xticks(range(len(hls))); ax.set_xticklabels(hls)
        ax.set_yticks(range(len(fls))); ax.set_yticklabels(fls)
        ax.set_xlabel('hist_len (ms)'); ax.set_ylabel('filter_len (ms)')
        ax.set_title(f'lambda = {lam:.0e}')
        plt.colorbar(im, ax=ax, label='worst val loss / spike')
    fig.suptitle('Grid search (black box = selected)')
    _finish(fig, outdir, 'cbem_gridsearch.pdf')


def plot_crossval(cv, outdir=None):
    names = list(cv)
    x, w = np.arange(len(names)), 0.27
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, [cv[n]['train_per_spk'] for n in names], w, label='train', color='steelblue', edgecolor='k')
    ax.bar(x, [cv[n]['val_per_spk'] for n in names], w, label='validation', color='salmon', edgecolor='k')
    ax.bar(x + w, [cv[n]['null_val_per_spk'] for n in names], w, label='null (constant rate)', color='lightgray', edgecolor='k')
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('negative log-likelihood per spike (lower = better)')
    ax.set_title('Cross-validation'); ax.legend()
    _finish(fig, outdir, 'cbem_crossval.pdf')


# ── filters ──────────────────────────────────────────────────────────────────
def plot_filters(results, dt=DT, outdir=None):
    names = list(results)
    fig, axes = plt.subplots(len(names), 3, figsize=(15, 4 * len(names)), squeeze=False)
    for row, n in enumerate(names):
        r = results[n]
        t = np.arange(len(r['k_e_mc'])) * dt * 1000
        a = axes[row]
        a[0].plot(t, r['k_e_mc'], color=BLUE, label='MC→ge'); a[0].plot(t, r['k_e_vc'], '--', color=BLUE, alpha=.6, label='VC→ge')
        a[0].plot(t, r['k_i_mc'], color=RED, label='MC→gi'); a[0].plot(t, r['k_i_vc'], '--', color=RED, alpha=.6, label='VC→gi')
        a[0].set_title(f'{n} — all filters')
        a[1].plot(t, r['k_e_mc'], color=BLUE, label='MC'); a[1].plot(t, r['k_e_vc'], color=RED, label='VC')
        a[1].set_title(f'{n} — excitatory (ge)')
        a[2].plot(t, r['k_i_mc'], color=BLUE, label='MC'); a[2].plot(t, r['k_i_vc'], color=RED, label='VC')
        a[2].set_title(f'{n} — inhibitory (gi)')
        for ax in a:
            ax.axhline(0, color='k', lw=.5); ax.set_xlabel('Lag (ms)'); ax.legend(fontsize=8)
    _finish(fig, outdir, 'cbem_filters.pdf')


def plot_filter_stats(table, outdir=None):
    names = list(dict.fromkeys(table['neuron']))
    labels = list(dict.fromkeys(table['filter']))
    cols = [BLUE, 'steelblue', RED, 'salmon']
    x, w = np.arange(len(names)), 0.2
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, col, title in ((axes[0], 'latency_ms', 'Peak latency (ms)'),
                           (axes[1], 'width_ms', 'Half-width, FWHM (ms)')):
        for j, lab in enumerate(labels):
            vals = [table[(table.neuron == n) & (table['filter'] == lab)][col].iloc[0] for n in names]
            ax.bar(x + (j - 1.5) * w, vals, w, label=lab, color=cols[j % 4], edgecolor='k')
        ax.set_xticks(x); ax.set_xticklabels(names); ax.set_title(title); ax.legend(fontsize=8)
    _finish(fig, outdir, 'cbem_filter_stats.pdf')


def plot_ei(ei, outdir=None):
    x, w = np.arange(len(ei)), 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (pw, e, i) in zip(axes, (('MC → neuron', 'mc_e', 'mc_i'), ('VC → neuron', 'vc_e', 'vc_i'))):
        ax.bar(x - w / 2, ei[e], w, label='excitatory', color=BLUE, edgecolor='k')
        ax.bar(x + w / 2, ei[i], w, label='inhibitory', color=RED, edgecolor='k')
        ax.set_xticks(x); ax.set_xticklabels(ei['neuron']); ax.set_title(pw)
        ax.set_ylabel('integrated positive filter weight'); ax.legend()
    _finish(fig, outdir, 'cbem_EI.pdf')


# ── single-protocol views ────────────────────────────────────────────────────
def plot_conductances(dec, title='', outdir=None):
    t = dec['t']
    fig, ax = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    ax[0].plot(t, dec['X'][:, 0], color=BLUE, label='MC laser')
    ax[0].plot(t, dec['X'][:, 1] + 1.2, color=RED, label='VC laser (offset)'); ax[0].legend()
    ax[0].set_title(title)
    ax[1].plot(t, dec['ge'], color=BLUE); ax[1].set_ylabel('ge (nS)')
    ax[2].plot(t, dec['gi'], color=RED); ax[2].set_ylabel('gi (nS)')
    ax[3].vlines(t[dec['spikes'] == 1], 0, 1, color='k', lw=.5)
    ax[3].set_ylabel('spikes'); ax[3].set_xlabel('Time (s)')
    _finish(fig, outdir, 'cbem_conductances.pdf')


def plot_decomposition(dec, title='', outdir=None):
    t = dec['t']
    fig, ax = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    ax[0].plot(t, dec['X'][:, 0], color=BLUE, label='MC')
    ax[0].plot(t, dec['X'][:, 1] + 1.1, color=RED, label='VC (offset)'); ax[0].legend(fontsize=8)
    ax[0].set_title(title)
    ax[1].plot(t, dec['V'], 'k'); ax[1].axhline(dec['vth'], color='orange', ls='--', label=f'Vth = {dec["vth"]:.1f} mV')
    ax[1].set_ylabel('V (mV)'); ax[1].legend(fontsize=8)
    ax[2].plot(t, dec['drive'], color=BLUE, label=f'a·(V−Vth)  [std {dec["std_drive"]:.2f}]')
    ax[2].axhline(0, color='k', lw=.4); ax[2].set_ylabel('stimulus-driven'); ax[2].legend(fontsize=8)
    ax[3].plot(t, dec['h_contrib'], color='darkorange', label=f'history  [std {dec["std_hist"]:.2f}]')
    ax[3].axhline(0, color='k', lw=.4); ax[3].set_ylabel('history'); ax[3].legend(fontsize=8)
    ax[4].vlines(t[dec['spikes'] == 1], 0, 1, color='k', lw=.5, label='spikes')
    ax[4].plot(t, dec['lam'] / DT, color='green', alpha=.7, label='predicted rate (Hz)')
    ax[4].set_xlabel('Time (s)'); ax[4].legend(fontsize=8)
    _finish(fig, outdir, 'cbem_decomposition.pdf')


# ── PSTH ─────────────────────────────────────────────────────────────────────
def plot_psth(psths, outdir=None):
    """psths: {(neuron, pathway): result-or-None}"""
    names = list(dict.fromkeys(k[0] for k in psths))
    paths = list(dict.fromkeys(k[1] for k in psths))
    colors = {'MC': BLUE, 'VC': RED}
    fig, axes = plt.subplots(len(names), len(paths), figsize=(5.5 * len(paths), 4 * len(names)), squeeze=False)
    for i, n in enumerate(names):
        for j, p in enumerate(paths):
            ax, r = axes[i, j], psths.get((n, p))
            ax.set_title(f'{n} — {p}')
            if r is None:
                ax.text(.5, .5, 'no valid trials', ha='center', transform=ax.transAxes); continue
            t = r['t_ms']
            ax.fill_between(t, r['obs_mean'] - r['obs_sem'], r['obs_mean'] + r['obs_sem'], color='k', alpha=.15)
            ax.plot(t, r['obs_mean'], 'k', label=f'observed (n={r["n_trials"]})')
            c = colors.get(p, 'green')
            ax.fill_between(t, r['pred_mean'] - r['pred_sem'], r['pred_mean'] + r['pred_sem'], color=c, alpha=.2)
            ax.plot(t, r['pred_mean'], '--', color=c, label='model')
            ax.axvline(0, color=GREY, ls=':'); ax.set_xlabel('Time from onset (ms)'); ax.set_ylabel('Rate (Hz)')
            ax.legend(fontsize=8)
    _finish(fig, outdir, 'cbem_psth.pdf')


# ── simultaneous condition ───────────────────────────────────────────────────
def plot_simultaneous(results, outdir=None):
    names = list(results)
    fig, axes = plt.subplots(len(names), 2, figsize=(14, 4 * len(names)), squeeze=False)
    for i, n in enumerate(names):
        s = results[n]['simul']
        t = np.arange(len(s['sim']['ge'])) * DT
        for ax, g in zip(axes[i], ('ge', 'gi')):
            ax.plot(t, s['mc'][g], color=BLUE, alpha=.7, label='MC only')
            ax.plot(t, s['vc'][g], color=RED, alpha=.7, label='VC only')
            ax.plot(t, s['lin'][g], '--', color=GREY, label='linear sum (baseline once)')
            ax.plot(t, s['sim'][g], 'k', label='simultaneous')
            ax.set_title(f'{n} — {g}'); ax.set_ylabel(f'{g} (nS)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=8)
    _finish(fig, outdir, 'cbem_simultaneous_conductances.pdf')


def plot_rate_comparison(ds, results, window, sigma_ms=20, outdir=None):
    a, b = window
    names = list(results)
    fig, axes = plt.subplots(len(names), 1, figsize=(14, 4 * len(names)), sharex=True, squeeze=False)
    for ax, n in zip(axes[:, 0], names):
        s = results[n]['simul']
        t = np.arange(b - a) * DT
        obs = gaussian_filter1d(ds.spk_1k[int(n.replace('neuron', ''))][a:b].astype(float), sigma_ms) / DT
        ax.plot(t, s['mc']['lam'] / DT, color=BLUE, alpha=.7, label='MC only (pred)')
        ax.plot(t, s['vc']['lam'] / DT, color=RED, alpha=.7, label='VC only (pred)')
        ax.plot(t, s['lin']['lam'] / DT, '--', color=GREY, label='linear sum (MC+VC)')
        ax.plot(t, s['sim']['lam'] / DT, 'k', label='simultaneous (pred)')
        ax.plot(t, obs, color='green', alpha=.6, label='observed (smoothed)')
        ax.set_title(f'{n} — predicted vs observed rate'); ax.set_ylabel('Hz'); ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel('Time (s)')
    _finish(fig, outdir, 'cbem_rate_comparison.pdf')


def plot_mi(mi, outdir=None):
    x, w = np.arange(len(mi)), 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, mi['MI_obs'], w, label='observed', color='steelblue', edgecolor='k')
    if 'MI_model' in mi:
        ax.bar(x + w / 2, mi['MI_model'], w, label='model', color='salmon', edgecolor='k')
    ax.axhline(0, color='k'); ax.set_xticks(x); ax.set_xticklabels(mi['neuron'])
    ax.set_ylabel('Modulation index'); ax.set_title('>0 super-additive | <0 sub-additive'); ax.legend()
    _finish(fig, outdir, 'cbem_MI.pdf')


def plot_summation(results, summ, outdir=None):
    names = list(results)
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 4 * len(names)), squeeze=False)
    for i, n in enumerate(names):
        s, row = results[n]['simul'], summ[summ.neuron == n].iloc[0]
        for ax, g, c in zip(axes[i], ('ge', 'gi'), (BLUE, RED)):
            base = s['base_e'] if g == 'ge' else s['base_i']
            x = row[f'alpha_{g}'] * (s['mc'][g] - base) + row[f'beta_{g}'] * (s['vc'][g] - base)
            y = s['sim'][g] - base
            ax.scatter(x, y, s=2, alpha=.3, color=c)
            lim = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lim, lim, 'k--', lw=1)
            ax.set_xlabel(f'α·Δ{g}_mc + β·Δ{g}_vc'); ax.set_ylabel(f'Δ{g}_sim')
            ax.set_title(f'{n} {g}: α={row["alpha_" + g]:.2f} β={row["beta_" + g]:.2f} R²={row["R2_" + g]:.2f}')
    _finish(fig, outdir, 'cbem_summation.pdf')
