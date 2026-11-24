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

Three ways to measure it (each gives an additivity index and a class):
  1. observed_modulation(raw, ...)   DATA ONLY: evoked rate, spike probability per
                                     pulse and first-spike latency (no model)
  2. model_modulation(params, ...)   MODEL ONLY: for every simultaneous pulse the
                                     model is run with both lasers, MC only and
                                     VC only; reports rate, ge and gi additivity
  3. combined_modulation(...)        BOTH: the model's prediction for the real
                                     MC / VC / Sim pulses is compared with the
                                     observed responses (cross-validation of the
                                     model's summation), plus a consensus call,
                                     the interaction the model does not explain,
                                     and the conductance mechanism
  modulation_table(...)  runs all three for every neuron of one file
  population_summary(df) pools neurons (per file and whole system)

Classes (additivity_class), for responses of the dominant sign:
  supra-additive  Sim > MC + VC
  linear          Sim = MC + VC (95 % CI of Sim - sum includes 0; model: |AI| < 0.1)
  sub-additive    max(MC, VC) <= Sim < MC + VC
  subtractive     Sim < max(MC, VC)   (adding the second input reduces the response)
  "(inhibitory)" is appended when MC + VC < 0: the same rules on the size of the
  suppression.
"""
import numpy as np

from .config import FS_ACQ

# protocol ids: (VC, MC, simultaneous)
PAIRS = {'single': (1, 3, 5), 'train': (2, 4, 6)}
# response window after each light onset (s); the same length is used before it
WINDOWS = {'single': 0.030, 'train': 0.100}


# ── helpers ──────────────────────────────────────────────────────────────────
# For trains only the FIRST pulse of each train is an onset: pulses inside a
# 100 Hz train are 10 ms apart, so a 100 ms window per pulse would put the
# previous pulses' response into the "pre" baseline and cancel the response.
MIN_GAP_S = 0.050


def light_onsets(stim, min_gap=0):
    """Indices where the laser goes 0 -> 1 (a block that starts with the laser
    already on counts its first sample as an onset). With min_gap > 0 (samples),
    an onset closer than min_gap to the previous one is dropped, so a train
    gives one onset."""
    s = np.asarray(stim).astype(np.int8)
    on = np.flatnonzero(np.diff(np.r_[0, s]) == 1)
    if min_gap > 0 and len(on):
        on = on[np.r_[True, np.diff(on) >= min_gap]]
    return on


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


def additivity_index(mc, vc, sim, min_resp=0.0):
    """(Sim - (MC + VC)) / (|MC| + |VC|): 0 linear, > 0 supra, < 0 sub, -1 = no
    response to Sim. Works for negative (inhibitory) responses. NaN when MC and
    VC are both below min_resp (the index would divide by ~0)."""
    norm = abs(mc) + abs(vc)
    return (sim - (mc + vc)) / norm if norm > max(min_resp, 0) and norm > 0 else np.nan


def additivity_class(mc, vc, sim, ci=None, tol=0.1, responsive=True, min_resp=0.0):
    """Supra-additive / linear / sub-additive / subtractive (see module doc).
    ci = (low, high) of Sim - sum: 'linear' if it includes 0. Without a CI
    (the model), 'linear' if |additivity index| < tol. min_resp: if MC, VC and
    Sim are all smaller than this, there is nothing to sum ('no response')."""
    if not responsive:
        return 'no response'
    if not np.all(np.isfinite([mc, vc, sim])):
        return 'n/a'
    if max(abs(mc), abs(vc), abs(sim)) <= min_resp:
        return 'no response'
    if abs(mc) + abs(vc) <= min_resp or abs(mc) + abs(vc) == 0:
        # only the combination responds: pure interaction, index undefined
        return ('supra-additive' if sim > 0 else 'supra-additive (inhibitory)') + ' (only Sim responds)'
    tag = ''
    if mc + vc < 0:   # suppression: classify the size of the suppression
        mc, vc, sim, tag = -mc, -vc, -sim, ' (inhibitory)'
        if ci is not None:
            ci = (-ci[1], -ci[0])
    d = sim - (mc + vc)
    linear = (ci[0] <= 0 <= ci[1]) if ci is not None else abs(d) / (abs(mc) + abs(vc)) < tol
    if linear:
        return 'linear' + tag
    if d > 0:
        return 'supra-additive' + tag
    return ('subtractive' if sim < max(mc, vc) else 'sub-additive') + tag


def spike_prob_per_pulse(spikes, onsets, win):
    """Per-pulse P(>= 1 spike) after (post) and before (pre) each onset."""
    ok = [o for o in onsets if o - win >= 0 and o + win <= len(spikes)]
    post = np.array([spikes[o:o + win].any() for o in ok], float)
    pre = np.array([spikes[o - win:o].any() for o in ok], float)
    return post, pre


def first_spike_latency_ms(spikes, onsets, win, fs):
    """Median and SD (jitter) of the first-spike latency over pulses that have a
    spike in the window, in ms."""
    lat = []
    for o in onsets:
        if o + win <= len(spikes):
            k = np.flatnonzero(spikes[o:o + win])
            if len(k):
                lat.append(k[0] / fs * 1000)
    if not lat:
        return np.nan, np.nan
    return float(np.median(lat)), float(np.std(lat))


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
        'Class': additivity_class(r_mc, r_vc, r_sim, (lo, hi), responsive=responsive),
        # point estimate, same rule as the model (|AI| < 0.1 = linear): used to
        # compare with the model, since the model has no CI
        'Class (point estimate)': additivity_class(r_mc, r_vc, r_sim),
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
    gap = int(MIN_GAP_S * fs)
    onsets = {p: light_onsets(stim[idx], gap) for p, idx in blocks.items()}
    light = {p: light_duration_ms(stim[idx], fs) for p, idx in blocks.items()}

    rows = []
    for n in neurons:
        spk = raw[:, n + 1].astype(np.int32)
        block_rate = {p: spk[idx].sum() / (len(idx) / fs) if len(idx) else np.nan
                      for p, idx in blocks.items()}
        med_rate = np.nanmedian(list(block_rate.values()))
        for kind, (p_vc, p_mc, p_sim) in PAIRS.items():
            win = int(windows[kind] * fs)
            d, prob, lat = {}, {}, {}
            for p in (p_vc, p_mc, p_sim):
                y = spk[blocks[p]]
                d[p], _ = evoked_per_pulse(y, onsets[p], win)
                post, pre = spike_prob_per_pulse(y, onsets[p], win)
                prob[p] = (post.mean() if len(post) else np.nan,
                           pre.mean() if len(pre) else np.nan)
                lat[p] = first_spike_latency_ms(y, onsets[p], win, fs)
            row = {'Neuron': n, 'Stimulus': kind}
            row.update(_summarise(d[p_mc], d[p_vc], d[p_sim], windows[kind], n_boot, rng))

            # spike probability per pulse. If baseline, MC and VC each made the
            # cell fire independently: 1 - P_sim = (1 - P_mc)(1 - P_vc) / (1 - P_base)
            # (dividing once by the no-spike-from-baseline term, which both
            # single conditions contain).
            pm, pv, ps = prob[p_mc][0], prob[p_vc][0], prob[p_sim][0]
            pb = np.nanmean([prob[p][1] for p in prob])
            p_ind = (1 - (1 - pm) * (1 - pv) / (1 - pb)) if pb < 1 else np.nan
            p_ind = float(np.clip(p_ind, 0, 1)) if np.isfinite(p_ind) else np.nan
            row.update({
                'P spike mc': pm, 'P spike vc': pv, 'P spike sim': ps,
                'P spike baseline': pb,
                'P spike independent': p_ind,
                'P sim - independent': ps - p_ind,
                'P spike class': ('n/a' if not np.isfinite(ps - p_ind) else
                                  'facilitation (> independent)' if ps - p_ind > 0.05 else
                                  'occlusion (< independent)' if ps - p_ind < -0.05 else
                                  'independent'),
                'Latency mc (ms)': lat[p_mc][0], 'Latency vc (ms)': lat[p_vc][0],
                'Latency sim (ms)': lat[p_sim][0],
                'Jitter mc/vc/sim (ms)': '/'.join(f'{lat[p][1]:.1f}' for p in (p_mc, p_vc, p_sim)),
                'Latency shift sim - fastest single (ms)':
                    lat[p_sim][0] - np.nanmin([lat[p_mc][0], lat[p_vc][0]])
                    if np.isfinite([lat[p_mc][0], lat[p_vc][0]]).any() else np.nan,
            })

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


# smallest model response treated as real: rate in Hz; conductances as a
# fraction of their light-on level
MODEL_MIN_RATE_HZ = 0.5
MODEL_MIN_G_FRAC = 0.01

# ── model ────────────────────────────────────────────────────────────────────
def _window_response(params, fl, hl, Xc, pad, win, dt, biophys, make_design_matrix, forward):
    """Evoked (post - pre) rate in Hz, ge and gi for one stimulus window, plus
    the post-window mean ge and gi. No spike history (stimulus-driven only)."""
    Xd = make_design_matrix(Xc, fl, [(0, len(Xc))])
    out = forward(params, Xd, None, fl, hl, dt, biophys)
    pre, post = slice(pad, pad + win), slice(pad + win, pad + 2 * win)
    lam = out['lam'] / dt
    return {'rate': lam[post].mean() - lam[pre].mean(),
            'ge': out['ge'][post].mean() - out['ge'][pre].mean(),
            'gi': out['gi'][post].mean() - out['gi'][pre].mean(),
            'ge_post': out['ge'][post].mean(), 'gi_post': out['gi'][post].mean()}


def _mean_resp(lst):
    keys = ('rate', 'ge', 'gi', 'ge_post', 'gi_post')
    return {k: float(np.mean([r[k] for r in lst])) if lst else np.nan for k in keys}


def model_modulation(params, fl, hl, X, protocol, dt, biophys, make_design_matrix,
                     forward, windows=WINDOWS):
    """MODEL ONLY. What the fitted CBEM predicts for the same comparison, with
    no spike history (stimulus-driven rate only).

    For every simultaneous pulse it runs the model three times on the same
    stretch of stimulus: both lasers, MC column only, VC column only. So the
    three conditions share timing and light duration exactly. Reports the
    additivity of the predicted rate and of the two conductances: ge (excitation)
    and gi (inhibition), and the inhibitory fraction gi / (ge + gi).

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
        on = idx[light_onsets(X[idx, 0], int(round(MIN_GAP_S / dt)))]
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
                resp[key].append(_window_response(params, fl, hl, Xc, pad, win, dt, biophys,
                                                  make_design_matrix, forward))
        r = {k: _mean_resp(v) for k, v in resp.items()}
        row = {'Stimulus': kind, 'n pulses': len(resp['sim'])}
        for q, label in (('rate', 'R'), ('ge', 'ge'), ('gi', 'gi')):
            mc, vc, sim = r['mc'][q], r['vc'][q], r['sim'][q]
            unit = ' (Hz)' if q == 'rate' else ''
            thr = (MODEL_MIN_RATE_HZ if q == 'rate' else
                   MODEL_MIN_G_FRAC * np.nanmax([r[k][q + '_post'] for k in r]))
            row.update({
                f'Model {label}_mc{unit}': mc, f'Model {label}_vc{unit}': vc,
                f'Model {label}_sim{unit}': sim,
                f'Model {label} additivity index': additivity_index(mc, vc, sim, thr),
                f'Model {label} class': additivity_class(mc, vc, sim, min_resp=thr),
            })
        # backwards-compatible names
        row['Model linear sum (Hz)'] = row['Model R_mc (Hz)'] + row['Model R_vc (Hz)']
        row['Model sim - sum (Hz)'] = row['Model R_sim (Hz)'] - row['Model linear sum (Hz)']
        row['Model additivity index'] = row['Model R additivity index']
        for k in ('mc', 'vc', 'sim'):
            ge, gi = r[k]['ge_post'], r[k]['gi_post']
            row[f'Model gi/(ge+gi) {k}'] = gi / (ge + gi) if ge + gi > 0 else np.nan
        row['Model mechanism'] = mechanism(row['Model ge class'], row['Model gi class'],
                                           row['Model R class'])
        rows.append(row)
    return rows


def _cond_text(cls, name, what):
    """e.g. ('subtractive', 'ge', 'excitation')"""
    inh = '(inhibitory)' in cls
    c = cls.split(' (')[0]
    dirn = f'{name} decreases at light onset' if inh else f'{name} increases'
    verb = {'sub-additive': 'saturates (sub-additive)',
            'subtractive': 'is smaller with both than with one (subtractive)',
            'supra-additive': 'grows more than the sum (supra-additive)',
            'linear': 'adds linearly'}.get(c)
    return f'{what}: {dirn}, {verb}' if verb else None


def mechanism(ge_class, gi_class, rate_class):
    """Plain-language reading of how the model produces the rate summation."""
    if rate_class in ('n/a', 'no response'):
        return 'no model response'
    parts = [t for t in (_cond_text(ge_class, 'ge', 'excitation'),
                         _cond_text(gi_class, 'gi', 'inhibition')) if t]
    if all('adds linearly' in t for t in parts):
        parts.append('rate nonlinearity comes from the voltage / spiking threshold')
    return '; '.join(parts) or 'n/a'


def model_predicted_conditions(params, fl, hl, X, protocol, dt, biophys,
                               make_design_matrix, forward, windows=WINDOWS):
    """The model's evoked rate (Hz, no spike history) for the REAL pulses of
    each protocol: VC only (1, 2), MC only (3, 4), simultaneous (5, 6). Same
    windows as observed_modulation, so the numbers compare one-to-one."""
    X = np.asarray(X, float)
    protocol = np.asarray(protocol).astype(int)
    out = {}
    for kind, (p_vc, p_mc, p_sim) in PAIRS.items():
        win = int(round(windows[kind] / dt))
        pad = fl
        for cond, p, col in (('vc', p_vc, 1), ('mc', p_mc, 0), ('sim', p_sim, 0)):
            idx = np.flatnonzero(protocol == p)
            if len(idx) == 0:
                out[(kind, cond)] = np.nan
                continue
            on = idx[light_onsets(X[idx, col], int(round(MIN_GAP_S / dt)))]
            vals = [_window_response(params, fl, hl, X[o - win - pad:o + win], pad, win, dt,
                                     biophys, make_design_matrix, forward)['rate']
                    for o in on if o - win - pad >= 0 and o + win <= len(X)]
            out[(kind, cond)] = float(np.mean(vals)) if vals else np.nan
    return out


# ── combined: model vs data ─────────────────────────────────────────────────
def combined_modulation(obs_row, model_row, pred, fit_quality=None):
    """BOTH. obs_row: one observed_modulation row; model_row: the matching
    model_modulation row; pred: model_predicted_conditions output (whole
    neuron, used for the cross-validation across all 6 conditions).

    Returns:
      Pred R_mc/vc/sim       model evoked rate on the real pulses
      Pred additivity index  the observed formula applied to the model prediction
      Agreement              observed class == predicted class (point estimates,
                             |AI| < 0.1 = linear for both)
      Consensus class        the class if both agree, else 'disagree'
      Combined AI            mean of observed and predicted AI (only if they agree)
      Unexplained interaction  observed AI - model (virtual) AI: the MC x VC
                               interaction the model's own nonlinearity cannot
                               produce
      Model vs data r / MAE  how well the model reproduces the 6 observed evoked
                             responses (single + train, MC / VC / Sim)
    """
    kind = obs_row['Stimulus']
    mc, vc, sim = (pred[(kind, c)] for c in ('mc', 'vc', 'sim'))
    ai_pred = additivity_index(mc, vc, sim, MODEL_MIN_RATE_HZ)
    cls_pred = additivity_class(mc, vc, sim, min_resp=MODEL_MIN_RATE_HZ)
    ai_obs = obs_row['Additivity index']
    if obs_row['Class'] == 'no response':
        # the data show no significant light response: nothing to cross-validate
        agree_txt, consensus, comb = 'n/a (no observed response)', 'no observed response', np.nan
    elif cls_pred in ('no response', 'n/a'):
        agree_txt, consensus, comb = 'n/a (no model response)', 'no model response', np.nan
    else:
        cls_obs = obs_row['Class (point estimate)']
        agree = cls_obs.split(' (')[0] == cls_pred.split(' (')[0]
        agree_txt = 'yes' if agree else 'no'
        consensus = cls_obs if agree else 'disagree'
        comb = np.nanmean([ai_obs, ai_pred]) if agree else np.nan
    row = {
        'Pred R_mc (Hz)': mc, 'Pred R_vc (Hz)': vc, 'Pred R_sim (Hz)': sim,
        'Pred additivity index': ai_pred, 'Pred class': cls_pred,
        'Agreement (obs vs model)': agree_txt,
        'Consensus class': consensus,
        'Combined AI': comb,
        'Unexplained interaction (obs AI - model AI)':
            ai_obs - model_row['Model R additivity index'],
    }
    if fit_quality is not None:
        row.update(fit_quality)
    return row


def prediction_quality(obs_rows, pred):
    """Correlation and mean absolute error between observed and predicted evoked
    rates over the 6 conditions (single/train x MC/VC/Sim) of one neuron."""
    o, p = [], []
    for r in obs_rows:
        for c, key in (('mc', 'R_mc (Hz)'), ('vc', 'R_vc (Hz)'), ('sim', 'R_sim (Hz)')):
            o.append(r[key]); p.append(pred[(r['Stimulus'], c)])
    o, p = np.array(o, float), np.array(p, float)
    ok = np.isfinite(o) & np.isfinite(p)
    r = np.corrcoef(o[ok], p[ok])[0, 1] if ok.sum() >= 3 and o[ok].std() > 0 and p[ok].std() > 0 else np.nan
    return {'Model vs data r (6 conditions)': r,
            'Model vs data MAE (Hz)': float(np.mean(np.abs(o[ok] - p[ok]))) if ok.any() else np.nan}


def modulation_table(raw, results, X, protocol, dt, biophys, make_design_matrix, forward,
                     neurons=None, fs=FS_ACQ, n_boot=2000):
    """All three methods for every neuron of one file, one row per (neuron,
    single/train). results: {'neuron1': {'params', 'fl', 'hl'}, ...}.
    Returns a pandas DataFrame."""
    import pandas as pd
    neurons = neurons or sorted(int(k.replace('neuron', '')) for k in results)
    obs = observed_modulation(raw, fs=fs, neurons=neurons, n_boot=n_boot)
    rows = []
    for n in neurons:
        res = results[f'neuron{n}']
        args = (res['params'], res['fl'], res['hl'], X, protocol, dt, biophys,
                make_design_matrix, forward)
        mod = {r['Stimulus']: r for r in model_modulation(*args)}
        pred = model_predicted_conditions(*args)
        o_rows = [r for r in obs if r['Neuron'] == n]
        q = prediction_quality(o_rows, pred)
        for o in o_rows:
            m = mod.get(o['Stimulus'], {})
            row = {f'OBS {k}': v for k, v in o.items() if k not in ('Neuron', 'Stimulus')}
            row = {'Neuron': n, 'Stimulus': o['Stimulus'], **row}
            row.update({f'MODEL {k}': v for k, v in m.items() if k != 'Stimulus'})
            if m:
                row.update({f'COMBINED {k}': v for k, v in
                            combined_modulation(o, m, pred, q).items()})
            rows.append(row)
    return pd.DataFrame(rows)


def _wilcoxon_p(x):
    from scipy.stats import wilcoxon
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x != 0)]
    return float(wilcoxon(x).pvalue) if len(x) >= 5 else np.nan


def population_summary(df, group_cols=('File', 'Stimulus')):
    """Pool neurons. One row per group (e.g. each file x stimulus, or the whole
    system with group_cols=('Stimulus',)): for each method the mean +/- SEM of
    the additivity index, a Wilcoxon test of AI = 0 (p < 0.05: summation is not
    linear across neurons), the class counts and the net result (majority class)."""
    import pandas as pd
    methods = {'Observed (data only)': ('OBS Additivity index', 'OBS Class'),
               'Model only': ('MODEL Model R additivity index', 'MODEL Model R class'),
               'Model ge (excitation)': ('MODEL Model ge additivity index', 'MODEL Model ge class'),
               'Model gi (inhibition)': ('MODEL Model gi additivity index', 'MODEL Model gi class'),
               'Model on real pulses': ('COMBINED Pred additivity index', 'COMBINED Pred class'),
               'Combined (consensus)': ('COMBINED Combined AI', 'COMBINED Consensus class'),
               'Spike probability (data only)': ('OBS P sim - independent', 'OBS P spike class')}
    out = []
    groups = df.groupby(list(group_cols)) if group_cols else [((), df)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        for name, (ai_col, cls_col) in methods.items():
            if ai_col not in g:
                continue
            cls = g[cls_col].fillna('n/a').str.split(' \\(').str[0]
            # AI only where that method found a response to sum
            ai = pd.to_numeric(g[ai_col], errors='coerce').where(
                ~cls.isin(['n/a', 'no response', 'no observed response', 'no model response']))
            counts = cls.value_counts()
            valid = counts.drop(['n/a', 'no response', 'disagree', 'no observed response',
                                 'no model response'], errors='ignore')
            net = valid.idxmax() if len(valid) else 'n/a'
            row = dict(zip(group_cols, key))
            row.update({
                'Method': name,
                'n neurons': int(g[[c for c in ('File', 'Neuron') if c in g]].drop_duplicates().shape[0]),
                'n with AI': int(ai.notna().sum()),
                'Mean AI': ai.mean(), 'SEM AI': ai.std() / np.sqrt(ai.notna().sum()) if ai.notna().sum() > 1 else np.nan,
                'Median AI': ai.median(), 'IQR AI': ai.quantile(0.75) - ai.quantile(0.25), 'Wilcoxon p (AI = 0)': _wilcoxon_p(ai),
                'Class counts': ', '.join(f'{k}: {v}' for k, v in counts.items()),
                'Net result': net,
            })
            if name == 'Combined (consensus)':
                a = g['COMBINED Agreement (obs vs model)']
                judged = a.isin(['yes', 'no'])
                row['Agreement obs vs model (%)'] = 100 * (a[judged] == 'yes').mean() if judged.any() else np.nan
                row['n rows judged (both responded)'] = int(judged.sum())
            out.append(row)
    return pd.DataFrame(out)


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
