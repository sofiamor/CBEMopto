"""Fitting, cross-validation and hyper-parameter search."""
import itertools
import pickle
import time

import numpy as np
from scipy.optimize import minimize

from .config import FAST_PASSES, FS_OUT, FULL_PASSES, MAXFUN, PASS_PRESETS
from .data import balanced_subset, make_design_matrix, train_val_segments
from .model import filters, make_p0, n_params, nll_segments

try:  # progress bars: tqdm.auto gives a widget in Jupyter, text in a terminal
    from tqdm.auto import tqdm
except ImportError:  # fall back to a plain one-line text bar
    tqdm = None


class _TextBar:
    """Minimal stand-in for tqdm when it is not installed."""

    def __init__(self, total, desc='', disable=False, **_):
        self.total, self.desc, self.n, self.disable = total, desc, 0, disable
        self.t0, self.post = time.perf_counter(), ''

    def update(self, k=1):
        self.n += k
        if self.disable:
            return
        el = time.perf_counter() - self.t0
        rate = el / self.n
        print(f'\r{self.desc}: {self.n}/{self.total} it | {_hms(el)} elapsed | '
              f'{rate:.1f}s/it | max left {_hms(rate * (self.total - self.n))} '
              f'| {self.post}   ', end='', flush=True)

    def set_postfix_str(self, s):
        self.post = s

    def close(self):
        if not self.disable:
            print()


def _hms(sec):
    sec = int(sec)
    return f'{sec // 3600}h{sec % 3600 // 60:02d}m' if sec >= 3600 else f'{sec // 60}m{sec % 60:02d}s'


def _bar(**kw):
    return tqdm(**kw) if tqdm is not None else _TextBar(**kw)


def resolve_passes(passes):
    """Accept a preset name ('quick', 'fast', 'full') or a list of
    (maxiter, ftol, gtol) tuples."""
    if isinstance(passes, str):
        if passes not in PASS_PRESETS:
            raise ValueError(f'Unknown preset {passes!r}; choose from {list(PASS_PRESETS)}')
        return PASS_PRESETS[passes]
    return list(passes)


def scale_lambda(lam, n_bins_search, n_bins_full):
    """The L2 penalty does not grow with data but the likelihood does, so a
    lambda tuned on a short subset is relatively too strong on the full set.
    Scale it to keep the penalty/likelihood ratio the same."""
    return lam * n_bins_full / n_bins_search


def fit_neuron(Xd, Y, segments, fl, hl, dt, biophys, lambda_l2=0.0,
               passes=FULL_PASSES, verbose=True, name='neuron', maxfun=MAXFUN,
               progress=True):
    """Multi-pass L-BFGS-B. L2 penalty acts on the stimulus filters only.

    passes:   preset name ('quick', 'fast', 'full') or list of (maxiter, ftol, gtol).
    maxfun:   max model evaluations per pass (None = no limit, see config.MAXFUN).
    progress: show a progress bar per pass. The bar counts iterations up to
              maxiter, so its time estimate is a WORST case: most passes stop
              earlier, when the loss stops improving (ftol). Watch 's/it' and
              the loss in the bar: once the loss barely changes, the pass is
              about to end.
    The result includes 'log': one entry per pass with loss, iterations,
    evaluations, seconds and the reason the pass stopped."""
    passes = resolve_passes(passes)
    Yf = Y.astype(float)
    last = {'loss': np.nan, 'evals': 0}

    def obj(p):
        v = (nll_segments(p, Xd, Yf, segments, fl, hl, dt, biophys)
             + lambda_l2 * np.sum(p[:4 * fl] ** 2))
        last['evals'] += 1
        last['loss'] = v if not np.isfinite(last['loss']) else min(last['loss'], v)
        return v

    p0 = make_p0(fl, hl)
    loss0 = obj(p0)
    best, log = None, []
    for i, (maxiter, ftol, gtol) in enumerate(passes):
        start = p0 if best is None else best.x
        opts = {'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                'maxfun': maxfun if maxfun is not None else 10**12}
        bar = _bar(total=maxiter, desc=f'{name} pass {i + 1}/{len(passes)}',
                   unit='it', disable=not progress, leave=True)
        ev0 = last['evals']

        def callback(xk):  # scipy calls this once per finished iteration
            bar.set_postfix_str(f'loss={last["loss"]:.2f} evals={last["evals"] - ev0}')
            bar.update(1)

        t0 = time.perf_counter()
        try:
            r = minimize(obj, start, method='L-BFGS-B', options=opts, callback=callback)
        finally:
            bar.close()
        secs = time.perf_counter() - t0
        log.append({'pass': i + 1, 'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                    'loss': float(r.fun), 'nit': int(r.nit), 'nfev': int(r.nfev),
                    'secs': secs, 'success': bool(r.success), 'message': str(r.message)})
        if best is None or r.fun < best.fun:
            best = r
        if verbose:
            print(f'  {name} pass {i + 1}: loss={r.fun:.2f} nit={r.nit} nfev={r.nfev} '
                  f'{secs:.0f}s | {r.message}')
    if verbose:
        print(f'  {name}: loss {loss0:.1f} -> {best.fun:.1f} '
              f'({sum(e["secs"] for e in log):.0f}s total)')
    return {
        'params': best.x,
        'neg_LL': nll_segments(best.x, Xd, Yf, segments, fl, hl, dt, biophys),
        'success': best.success,
        'fl': fl, 'hl': hl, 'lambda_l2': lambda_l2,
        'loss0': float(loss0), 'log': log,
    }


def compare_passes(Xd, Y, segments, fl, hl, dt, biophys, settings,
                   lambda_l2=0.0, maxfun=MAXFUN, name='neuron'):
    """Fit ONE neuron with several optimizer settings and print a comparison,
    to see whether more iterations actually lower the loss or only take longer.

    settings: dict {label: preset name or list of (maxiter, ftol, gtol)}, e.g.
        {'quick': 'quick',
         '500 it': [(500, 1e-6, 1e-5)],
         'full': 'full'}
    Tip: run it on a short subset first; each fit costs ~(n_params + 1)
    model evaluations per iteration."""
    print(f'{name}: {n_params(fl, hl)} parameters, '
          f'{sum(b - a for a, b in segments)} bins')
    rows = {}
    for label, passes in settings.items():
        res = fit_neuron(Xd, Y, segments, fl, hl, dt, biophys, lambda_l2,
                         passes, verbose=False, name=name, maxfun=maxfun)
        log = res['log']
        rows[label] = {'loss': res['neg_LL'],
                       'nit': sum(e['nit'] for e in log),
                       'nfev': sum(e['nfev'] for e in log),
                       'secs': sum(e['secs'] for e in log),
                       'stopped': log[-1]['message'], 'params': res['params']}
    best_loss = min(r['loss'] for r in rows.values())
    print(f'\n{"setting":<14}{"neg_LL":>12}{"vs best":>10}{"iters":>8}'
          f'{"evals":>10}{"time":>9}  last pass stopped because')
    for label, r in rows.items():
        print(f'{label:<14}{r["loss"]:>12.2f}{r["loss"] - best_loss:>10.2f}'
              f'{r["nit"]:>8}{r["nfev"]:>10}{r["secs"]:>8.0f}s  {r["stopped"]}')
    print('\n"vs best" = extra loss compared with the best setting. '
          'Differences below ~1 are negligible.')
    return rows


def fit_all(fit, Xd, fl, hl, dt, biophys, lambda_l2=0.0, passes=FULL_PASSES,
            progress=True):
    """Fit every neuron. With progress=True an overall bar counts neurons; after
    the first neuron its time estimate is a fair guess for the whole file."""
    results = {}
    names = list(fit.spikes)
    overall = _bar(total=len(names), desc='neurons', unit='neuron',
                   disable=not progress or len(names) < 2, leave=True)
    try:
        for name in names:
            Y = fit.spikes[name]
            print(f'Fitting {name} ({int(Y.sum())} spikes, {Y.sum() / (len(Y) * dt):.1f} Hz)')
            res = fit_neuron(Xd, Y, fit.segments, fl, hl, dt, biophys,
                             lambda_l2, passes, name=name, progress=progress)
            res.update(filters(res['params'], fl, hl))
            results[name] = res
            overall.update(1)
    finally:
        overall.close()
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
                                 passes, verbose=False, name=n, progress=False)
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