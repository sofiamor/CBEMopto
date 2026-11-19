"""Fitting, cross-validation and hyper-parameter search."""
import itertools
import pickle

import numpy as np
from scipy.optimize import minimize

from .config import FAST_PASSES, FS_OUT, FULL_PASSES
from .data import balanced_subset, make_design_matrix, train_val_segments
from .model import filters, make_p0, nll_segments


def scale_lambda(lam, n_bins_search, n_bins_full):
    """The L2 penalty does not grow with data but the likelihood does, so a
    lambda tuned on a short subset is relatively too strong on the full set.
    Scale it to keep the penalty/likelihood ratio the same."""
    return lam * n_bins_full / n_bins_search


def fit_neuron(Xd, Y, segments, fl, hl, dt, biophys, lambda_l2=0.0,
               passes=FULL_PASSES, verbose=True, name='neuron'):
    """Multi-pass L-BFGS-B. L2 penalty acts on the stimulus filters only."""
    Yf = Y.astype(float)

    def obj(p):
        return (nll_segments(p, Xd, Yf, segments, fl, hl, dt, biophys)
                + lambda_l2 * np.sum(p[:4 * fl] ** 2))

    p0 = make_p0(fl, hl)
    loss0 = obj(p0)
    best = None
    for i, (maxiter, ftol, gtol) in enumerate(passes):
        start = p0 if best is None else best.x
        r = minimize(obj, start, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        if best is None or r.fun < best.fun:
            best = r
        if verbose:
            print(f'  {name} pass {i + 1}: loss={r.fun:.2f} nit={r.nit} ok={r.success}')
    if verbose:
        print(f'  {name}: loss {loss0:.1f} -> {best.fun:.1f}')
    return {
        'params': best.x,
        'neg_LL': nll_segments(best.x, Xd, Yf, segments, fl, hl, dt, biophys),
        'success': best.success,
        'fl': fl, 'hl': hl, 'lambda_l2': lambda_l2,
    }


def fit_all(fit, Xd, fl, hl, dt, biophys, lambda_l2=0.0, passes=FULL_PASSES):
    results = {}
    for name, Y in fit.spikes.items():
        print(f'Fitting {name} ({int(Y.sum())} spikes, {Y.sum() / (len(Y) * dt):.1f} Hz)')
        res = fit_neuron(Xd, Y, fit.segments, fl, hl, dt, biophys,
                         lambda_l2, passes, name=name)
        res.update(filters(res['params'], fl, hl))
        results[name] = res
    return results


def save_results(results, path):
    with open(path, 'wb') as f:
        pickle.dump(results, f)


def load_results(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ── cross-validation ─────────────────────────────────────────────────────────
def cross_validate(fit, Xd, fl, hl, dt, biophys, lambda_l2=0.0,
                   val_frac=0.2, passes=FULL_PASSES):
    """Refit on the first (1-val_frac) of every segment, evaluate on the rest.
    Losses are per spike so train, validation and the constant-rate null model
    are directly comparable."""
    tr, vl = train_val_segments(fit.segments, val_frac)
    if not vl:
        raise ValueError('Segments too short to hold out a validation part')
    n_tr_bins = sum(b - a for a, b in tr)
    n_vl_bins = sum(b - a for a, b in vl)
    out = {}
    for name, Y in fit.spikes.items():
        Yf = Y.astype(float)
        n_tr_spk = sum(Yf[a:b].sum() for a, b in tr)
        n_vl_spk = sum(Yf[a:b].sum() for a, b in vl)
        res = fit_neuron(Xd, Y, tr, fl, hl, dt, biophys, lambda_l2,
                         passes, verbose=False, name=name)
        val_nll = nll_segments(res['params'], Xd, Yf, vl, fl, hl, dt, biophys)
        p = np.clip(n_tr_spk / n_tr_bins, 1e-10, 1 - 1e-10)
        null_nll = -(n_vl_spk * np.log(p) + (n_vl_bins - n_vl_spk) * np.log1p(-p))
        out[name] = {
            'train_per_spk': res['neg_LL'] / max(n_tr_spk, 1),
            'val_per_spk': val_nll / max(n_vl_spk, 1),
            'null_val_per_spk': null_nll / max(n_vl_spk, 1),
            'n_train_spk': int(n_tr_spk), 'n_val_spk': int(n_vl_spk),
            'success': res['success'],
        }
        print(f'{name}: train {out[name]["train_per_spk"]:.3f} | '
              f'val {out[name]["val_per_spk"]:.3f} | '
              f'null {out[name]["null_val_per_spk"]:.3f}  (nats/spike)')
    return out


# ── hyper-parameter grid search ──────────────────────────────────────────────
def grid_search(fit, grid, dt, biophys, search_neurons=None, search_secs=60,
                val_frac=0.2, passes=FAST_PASSES, min_val_spikes=5, verbose=True):
    """Sweep filter_len x hist_len x lambda_l2 on a protocol-balanced subset.
    Candidates are ranked by their WORST validation loss per spike across
    neurons, so the winner works acceptably for all of them.

    The returned best['lambda'] is on the subset's scale; convert it with
    scale_lambda(best['lambda'], best['n_train_bins'], fit.T) before the full fit."""
    names = list(search_neurons or fit.spikes.keys())
    sub = balanced_subset(fit, int(search_secs * FS_OUT))
    tr, vl = train_val_segments(sub.segments, val_frac)
    n_tr_bins = sum(b - a for a, b in tr)
    n_vl_spk = {n: sum(sub.spikes[n][a:b].sum() for a, b in vl) for n in names}
    eligible = [n for n in names if n_vl_spk[n] >= min_val_spikes]
    for n in set(names) - set(eligible):
        print(f'{n}: skipped (only {int(n_vl_spk[n])} validation spikes in subset)')
    if not eligible:
        raise ValueError('No neuron has enough validation spikes; raise search_secs')

    n_combo = len(grid['filter_len']) * len(grid['hist_len']) * len(grid['lambda_l2'])
    print(f'{n_combo} combinations x {len(eligible)} neurons | subset {sub.T} bins, '
          f'train {n_tr_bins} bins')
    records = []
    for fl in grid['filter_len']:
        Xd = make_design_matrix(sub.X, fl, sub.segments)
        for hl, lam in itertools.product(grid['hist_len'], grid['lambda_l2']):
            per = {}
            for n in eligible:
                res = fit_neuron(Xd, sub.spikes[n], tr, fl, hl, dt, biophys, lam,
                                 passes, verbose=False, name=n)
                val = nll_segments(res['params'], Xd, sub.spikes[n].astype(float),
                                   vl, fl, hl, dt, biophys)
                per[n] = val / n_vl_spk[n]
            rec = {'fl': fl, 'hl': hl, 'lambda': lam,
                   'worst_val_per_spk': max(per.values()),
                   'per_neuron': per, 'n_train_bins': n_tr_bins}
            records.append(rec)
            if verbose:
                print(f'fl={fl:3d} hl={hl:2d} lam={lam:.0e} | worst={rec["worst_val_per_spk"]:.4f} | '
                      + ' '.join(f'{k}={v:.4f}' for k, v in per.items()))
    best = min(records, key=lambda r: r['worst_val_per_spk'])
    print(f'\nBEST: filter_len={best["fl"]} hist_len={best["hl"]} '
          f'lambda(subset scale)={best["lambda"]:.0e} '
          f'worst val/spk={best["worst_val_per_spk"]:.4f}')
    return records, best
