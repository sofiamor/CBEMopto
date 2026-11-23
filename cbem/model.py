"""CBEM forward model: stimulus -> conductances -> voltage -> spike probability.

Parameter vector layout (fl = filter_len, hl = hist_len):
  [0      : 2fl]      k_e   excitatory stimulus filters (MC then VC)
  [2fl    : 4fl]      k_i   inhibitory stimulus filters (MC then VC)
  [4fl    : 4fl+hl]   h     post-spike history filter
  [4fl+hl]            dc_e  baseline excitatory offset
  [4fl+hl+1]          dc_i  baseline inhibitory offset
  [4fl+hl+2]          a     gain
  [4fl+hl+3]          Vth   threshold (mV)
"""
import numpy as np

from .config import ODE_CONDUCTANCE_LAG

try:
    import numba
    HAS_NUMBA = True
except ImportError:  # pure-numpy fallback is slower but identical
    HAS_NUMBA = False


def softplus(x):
    return np.logaddexp(0, x) 


def n_params(fl, hl):
    return 4 * fl + hl + 4


def make_p0(fl, hl):
    p0 = np.zeros(n_params(fl, hl))
    p0[4 * fl + hl] = -3.0        # dc_e
    p0[4 * fl + hl + 1] = -3.0    # dc_i
    p0[4 * fl + hl + 2] = 0.5     # a
    p0[4 * fl + hl + 3] = -55.0   # Vth
    return p0


def unpack(params, fl, hl):
    return {
        'k_e': params[:2 * fl],
        'k_i': params[2 * fl:4 * fl],
        'h': params[4 * fl:4 * fl + hl],
        'dc_e': params[4 * fl + hl],
        'dc_i': params[4 * fl + hl + 1],
        'a': params[4 * fl + hl + 2],
        'vth': params[4 * fl + hl + 3],
    }


def filters(params, fl, hl):
    p = unpack(params, fl, hl)
    return {'k_e_mc': p['k_e'][:fl], 'k_e_vc': p['k_e'][fl:],
            'k_i_mc': p['k_i'][:fl], 'k_i_vc': p['k_i'][fl:], 'h': p['h']}


# ── voltage ODE (exponential Euler) ──────────────────────────────────────────
def _euler_numpy(ge, gi, El, Ee, Ei, gl, Cm, V0, dt, lag):
    g_tot = gl + ge + gi
    V_inf = (gl * El + ge * Ee + gi * Ei) / g_tot
    alpha = np.exp(-dt * g_tot / Cm)
    V = np.empty(len(ge))
    V[0] = V0
    for t in range(1, len(ge)):
        s = t - lag
        V[t] = alpha[s] * V[t - 1] + (1.0 - alpha[s]) * V_inf[s]
    return V


if HAS_NUMBA:
    _euler_numba = numba.njit(cache=True)(_euler_numpy)


def integrate_voltage(ge, gi, biophys, dt, lag=ODE_CONDUCTANCE_LAG):
    f = _euler_numba if HAS_NUMBA else _euler_numpy
    return f(ge, gi, biophys['El'], biophys['Ee'], biophys['Ei'],
             biophys['gl'], biophys['Cm'], biophys['V0'], dt, lag)


# ── spike history ────────────────────────────────────────────────────────────
def history_contribution(Y, h):
    """sum_lag h[lag] * Y[t - 1 - lag]  (strictly causal)."""
    hl, T = len(h), len(Y)
    Ypad = np.concatenate([np.zeros(hl), Y])
    out = np.zeros(T)
    for lag in range(hl):
        out += h[lag] * Ypad[hl - 1 - lag: T + hl - 1 - lag]
    return out


# ── forward pass on ONE contiguous segment ───────────────────────────────────
def forward(params, Xd, Y, fl, hl, dt, biophys):
    """Y = observed spikes for the history term, or None for zero history
    (stimulus-driven prediction only)."""
    p = unpack(params, fl, hl)
    ge = softplus(Xd @ p['k_e'] + p['dc_e'])
    gi = softplus(Xd @ p['k_i'] + p['dc_i'])
    V = integrate_voltage(ge, gi, biophys, dt)
    h_contrib = history_contribution(Y, p['h']) if Y is not None else np.zeros(len(V))
    drive = p['a'] * (V - p['vth'])
    lam = np.clip(softplus(drive + h_contrib) * dt, 1e-10, 1.0 - 1e-10)
    return {'ge': ge, 'gi': gi, 'V': V, 'drive': drive,
            'h_contrib': h_contrib, 'lam': lam}


def nll(params, Xd, Y, fl, hl, dt, biophys):
    lam = forward(params, Xd, Y, fl, hl, dt, biophys)['lam']
    return -np.sum(Y * np.log(lam) + (1 - Y) * np.log1p(-lam))


def nll_segments(params, Xd, Y, segments, fl, hl, dt, biophys):
    """Negative log-likelihood summed over segments. Each segment restarts the
    voltage at V0 and the spike history at zero."""
    return sum(nll(params, Xd[a:b], Y[a:b], fl, hl, dt, biophys)
               for a, b in segments)


# ── analytic gradient (adjoint through the exponential-Euler recursion) ─────
def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _euler_adjoint_numpy(gV, V, alpha, V_inf, lag):
    """Back-propagate dL/dV through V[t] = alpha[s] V[t-1] + (1-alpha[s]) V_inf[s],
    s = t - lag. Returns dL/dalpha and dL/dV_inf (indexed like alpha / V_inf)."""
    T = len(V)
    g_alpha = np.zeros(T)
    g_vinf = np.zeros(T)
    adj = 0.0
    for t in range(T - 1, 0, -1):
        adj = adj + gV[t]                      # total dL/dV[t]
        s = t - lag
        g_alpha[s] += adj * (V[t - 1] - V_inf[s])
        g_vinf[s] += adj * (1.0 - alpha[s])
        adj = adj * alpha[s]                   # carried back to V[t-1]
    return g_alpha, g_vinf


if HAS_NUMBA:
    _euler_adjoint = numba.njit(cache=True)(_euler_adjoint_numpy)
else:
    _euler_adjoint = _euler_adjoint_numpy


def nll_and_grad(params, Xd, Y, fl, hl, dt, biophys, lag=ODE_CONDUCTANCE_LAG):
    """Negative log-likelihood of ONE segment and its exact gradient."""
    p = unpack(params, fl, hl)
    El, Ee, Ei = biophys['El'], biophys['Ee'], biophys['Ei']
    gl, Cm = biophys['gl'], biophys['Cm']

    ue = Xd @ p['k_e'] + p['dc_e']
    ui = Xd @ p['k_i'] + p['dc_i']
    ge, gi = softplus(ue), softplus(ui)
    g_tot = gl + ge + gi
    V_inf = (gl * El + ge * Ee + gi * Ei) / g_tot
    alpha = np.exp(-dt * g_tot / Cm)
    V = integrate_voltage(ge, gi, biophys, dt, lag)

    h_contrib = history_contribution(Y, p['h'])
    z = p['a'] * (V - p['vth']) + h_contrib
    lam_raw = softplus(z) * dt
    lam = np.clip(lam_raw, 1e-10, 1.0 - 1e-10)
    loss = -np.sum(Y * np.log(lam) + (1 - Y) * np.log1p(-lam))

    inside = (lam_raw > 1e-10) & (lam_raw < 1.0 - 1e-10)
    g_lam = np.where(inside, -Y / lam + (1 - Y) / (1 - lam), 0.0)
    g_z = g_lam * _sigmoid(z) * dt

    grad = np.zeros_like(params, dtype=float)
    o = 4 * fl + hl
    grad[o + 2] = np.sum(g_z * (V - p['vth']))          # a
    grad[o + 3] = -p['a'] * np.sum(g_z)                 # Vth
    T = len(Y)
    Ypad = np.concatenate([np.zeros(hl), Y])
    for l in range(hl):                                 # history filter
        grad[4 * fl + l] = g_z @ Ypad[hl - 1 - l: T + hl - 1 - l]

    g_alpha, g_vinf = _euler_adjoint(p['a'] * g_z, V, alpha, V_inf, lag)
    g_ge = g_alpha * alpha * (-dt / Cm) + g_vinf * (Ee - V_inf) / g_tot
    g_gi = g_alpha * alpha * (-dt / Cm) + g_vinf * (Ei - V_inf) / g_tot
    g_ue = g_ge * _sigmoid(ue)
    g_ui = g_gi * _sigmoid(ui)
    grad[:2 * fl] = Xd.T @ g_ue
    grad[2 * fl:4 * fl] = Xd.T @ g_ui
    grad[o] = g_ue.sum()
    grad[o + 1] = g_ui.sum()
    return loss, grad


def nll_and_grad_segments(params, Xd, Y, segments, fl, hl, dt, biophys):
    loss, grad = 0.0, np.zeros(len(params))
    for a, b in segments:
        l, g = nll_and_grad(params, Xd[a:b], Y[a:b], fl, hl, dt, biophys)
        loss += l
        grad += g
    return loss, grad
