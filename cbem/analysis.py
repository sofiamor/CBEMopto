"""Post-fit analyses: filter statistics, E/I, decomposition, PSTH,
simultaneous (MC+VC) predictions, modulation index and summation tests."""
import warnings

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.signal import peak_widths

from .config import DT, FS_OUT
from .data import make_design_matrix, runs
from .model import forward, softplus, unpack

_trapz = getattr(np, 'trapezoid', None) or np.trapz

FILTER_KEYS = ['k_e_mc', 'k_e_vc', 'k_i_mc', 'k_i_vc']
FILTER_LABELS = ['MC→ge', 'VC→ge', 'MC→gi', 'VC→gi']


# ── D. filter peak latency / width ───────────────────────────────────────────
def filter_stats(filt, dt_ms=1.0):
    """(peak latency ms, FWHM ms, sign) of the dominant lobe (largest |value|).
    A negative dominant lobe is analysed on the sign-flipped filter."""
    filt = np.asarray(filt, float)
    i = int(np.argmax(np.abs(filt)))
    sign = 1.0 if filt[i] >= 0 else -1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            width = float(peak_widths(sign * filt, [i], rel_height=0.5)[0][0]) * dt_ms
    except Exception:
        width = float('nan')
    return i * dt_ms, width, sign


def filter_table(results, dt=DT):
    rows = []
    for name, r in results.items():
        for key, label in zip(FILTER_KEYS, FILTER_LABELS):
            lat, wid, sign = filter_stats(r[key], dt * 1000)
            rows.append({'neuron': name, 'filter': label, 'latency_ms': lat,
                         'width_ms': wid, 'sign': '+' if sign > 0 else '−'})
    return pd.DataFrame(rows)


def ei_table(results, dt=DT):
    """Integrated positive filter weight per pathway and E/I ratio."""
    rows = []
    for name, r in results.items():
        t = np.arange(len(r['k_e_mc'])) * dt * 1000
        ig = lambda k: _trapz(np.maximum(r[k], 0), t)
        mc_e, mc_i, vc_e, vc_i = ig('k_e_mc'), ig('k_i_mc'), ig('k_e_vc'), ig('k_i_vc')
        rows.append({'neuron': name, 'mc_e': mc_e, 'mc_i': mc_i, 'vc_e': vc_e,
                     'vc_i': vc_i, 'MC_EI': mc_e / (mc_i + 1e-6),
                     'VC_EI': vc_e / (vc_i + 1e-6)})
    return pd.DataFrame(rows)


# ── C. voltage / history decomposition ───────────────────────────────────────
def _first_run(ds, protocol):
    segs = [r for r in runs(ds.protocol_1k) if ds.protocol_1k[r[0]] == protocol]
    if not segs:
        raise ValueError(f'Protocol {protocol} not in data')
    return segs[0]


def decompose(ds, params, nidx, protocol, fl, hl, dt, biophys,
              start_bin=0, n_bins=5000):
    """Simulate from the start of the protocol's first run (so the voltage has
    its true history), then return the window [start_bin, start_bin + n_bins)."""
    a, b = _first_run(ds, protocol)
    end = min(b, a + start_bin + n_bins)
    X = ds.X_full[a:end]
    Y = ds.spk_1k[nidx][a:end].astype(float)
    out = forward(params, make_design_matrix(X, fl), Y, fl, hl, dt, biophys)
    sl = slice(start_bin, end - a)
    res = {k: v[sl] for k, v in out.items()}
    res.update(t=np.arange(len(res['V'])) * dt, X=X[sl], spikes=Y[sl],
               vth=float(unpack(params, fl, hl)['vth']),
               std_drive=float(np.std(res['drive'])),
               std_hist=float(np.std(res['h_contrib'])))
    return res


# ── B. PSTH vs predicted rate ────────────────────────────────────────────────
def psth(ds, params, nidx, fl, hl, dt, biophys, channel, protocols,
         pre_ms=20, post_ms=80, min_gap_ms=50):
    """Align observed spikes and the model's stimulus-driven rate to pulse
    onsets. For trains only the FIRST pulse of a train counts as an onset
    (a pulse counts if no other onset precedes it within min_gap_ms).
    The model is simulated over the whole run, then windows are cropped, so
    the voltage carries its real history. Returns None if no valid trial."""
    chan = ds.X_full[:, channel]
    spk = ds.spk_1k[nidx].astype(float)
    pre = int(round(pre_ms * FS_OUT / 1000))
    post = int(round(post_ms * FS_OUT / 1000))
    gap = int(round(min_gap_ms * FS_OUT / 1000))
    obs, pred = [], []
    for a, b in runs(ds.protocol_1k):
        if ds.protocol_1k[a] not in protocols:
            continue
        edges = np.flatnonzero(np.diff(chan[a:b]) > 0.5) + 1
        keep = [e for i, e in enumerate(edges) if i == 0 or e - edges[i - 1] >= gap]
        keep = [e for e in keep if e - pre >= 0 and e + post <= b - a]
        if not keep:
            continue
        lam = forward(params, make_design_matrix(ds.X_full[a:b], fl), None,
                      fl, hl, dt, biophys)['lam'] / dt
        for e in keep:
            obs.append(spk[a + e - pre: a + e + post] / dt)
            pred.append(lam[e - pre: e + post])
    if not obs:
        return None
    obs, pred = np.array(obs), np.array(pred)
    n = len(obs)
    return {'t_ms': np.arange(-pre, post) * 1000.0 / FS_OUT, 'n_trials': n,
            'obs_mean': obs.mean(0), 'obs_sem': obs.std(0) / np.sqrt(n),
            'pred_mean': pred.mean(0), 'pred_sem': pred.std(0) / np.sqrt(n)}


# ── simultaneous MC+VC predictions ───────────────────────────────────────────
def simultaneous_predictions(ds, results, fl, hl, dt, biophys, protocol=5, n_sec=10):
    """Simulate MC-only, VC-only and simultaneous stimulation on the first
    n_sec of a simultaneous run. Stores results[name]['simul'] and returns
    the (start, stop) window in the 1 kHz arrays.

    The linear conductance prediction is baseline-corrected:
        ge_lin = base + (ge_mc - base) + (ge_vc - base),  base = softplus(dc_e)
    so the baseline is counted once, not twice."""
    a, b = _first_run(ds, protocol)
    b = min(b, a + int(n_sec * FS_OUT))
    X_sim = ds.X_full[a:b]
    X_mc = X_sim.copy(); X_mc[:, 1] = 0
    X_vc = X_sim.copy(); X_vc[:, 0] = 0
    for name, r in results.items():
        p = unpack(r['params'], fl, hl)
        base_e, base_i = float(softplus(p['dc_e'])), float(softplus(p['dc_i']))
        sim = {}
        for tag, X in (('mc', X_mc), ('vc', X_vc), ('sim', X_sim)):
            o = forward(r['params'], make_design_matrix(X, fl), None, fl, hl, dt, biophys)
            sim[tag] = {'ge': o['ge'], 'gi': o['gi'], 'lam': o['lam']}
        sim['lin'] = {
            'ge': base_e + (sim['mc']['ge'] - base_e) + (sim['vc']['ge'] - base_e),
            'gi': base_i + (sim['mc']['gi'] - base_i) + (sim['vc']['gi'] - base_i),
            'lam': sim['mc']['lam'] + sim['vc']['lam'],
        }
        sim['base_e'], sim['base_i'] = base_e, base_i
        r['simul'] = sim
    return a, b


def modulation_index(ds, results, protocols=(1, 3, 5)):
    """MI = (sim - (mc + vc)) / (sim + mc + vc); >0 super-additive.
    protocols = (VC-only, MC-only, simultaneous). Observed MI uses the mean
    spike rate of each protocol; model MI uses the simulated window."""
    p_vc, p_mc, p_sim = protocols
    rows = []
    for name, r in results.items():
        spk = ds.spk_1k[int(name.replace('neuron', ''))]
        rate = lambda p: spk[ds.protocol_1k == p].mean() * FS_OUT
        r_mc, r_vc, r_sim = rate(p_mc), rate(p_vc), rate(p_sim)
        mi_obs = (r_sim - (r_mc + r_vc)) / (r_sim + r_mc + r_vc + 1e-10)
        row = {'neuron': name, 'r_mc_Hz': r_mc, 'r_vc_Hz': r_vc,
               'r_sim_Hz': r_sim, 'MI_obs': mi_obs}
        if 'simul' in r:
            m = {k: r['simul'][k]['lam'].mean() for k in ('mc', 'vc', 'sim')}
            row['MI_model'] = (m['sim'] - (m['mc'] + m['vc'])) / (m['sim'] + m['mc'] + m['vc'] + 1e-10)
        rows.append(row)
    return pd.DataFrame(rows)


def summation_test(results):
    """Regress the simultaneous conductance change on the single-pathway
    changes (baseline removed, no intercept):  d_sim ~ alpha*d_mc + beta*d_vc.
    alpha = beta = 1 -> linear; <1 -> sublinear; >1 -> supralinear."""
    rows = []
    for name, r in results.items():
        s = r['simul']
        row = {'neuron': name}
        for g, base in (('ge', s['base_e']), ('gi', s['base_i'])):
            A = np.stack([s['mc'][g] - base, s['vc'][g] - base], axis=1)
            y = s['sim'][g] - base
            coef = lstsq(A, y, rcond=None)[0]
            ss_res = np.sum((y - A @ coef) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            row.update({f'alpha_{g}': coef[0], f'beta_{g}': coef[1],
                        f'R2_{g}': 1 - ss_res / (ss_tot + 1e-12)})
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(results, cv=None, mi=None, summ=None):
    df = ei_table(results)[['neuron', 'MC_EI', 'VC_EI']]
    for extra in (mi, summ):
        if extra is not None:
            df = df.merge(extra, on='neuron')
    if cv is not None:
        c = pd.DataFrame(cv).T[['val_per_spk', 'null_val_per_spk']]
        c.index.name = 'neuron'
        df = df.merge(c.reset_index(), on='neuron')
    return df
