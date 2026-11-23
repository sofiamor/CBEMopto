"""Constants and default settings for the CBEM pipeline."""

FS_ACQ = 30000                    # acquisition sample rate (Hz)
FS_OUT = 1000                     # model sample rate (Hz)
DOWNSAMPLE = FS_ACQ // FS_OUT     # samples per model bin
DT = 1.0 / FS_OUT                 # model time step (s)

PROTOCOL_NAMES = {
    1: 'VC single pulses',
    2: 'VC 100Hz trains',
    3: 'MC single pulses',
    4: 'MC 100Hz trains',
    5: 'Simultaneous single pulses',
    6: 'Simultaneous 100Hz trains',
}

# Update from your patch data (see README cell in the notebook)
BIOPHYS = {
    'El': -65.0,    # mV  leak reversal / resting potential
    'Ee':   0.0,    # mV  excitatory reversal
    'Ei': -55.0,    # mV  inhibitory reversal
    'gl':   0.9,    # nS  leak conductance  (1 / Rin)
    'Cm':   0.99,   # nF  membrane capacitance (tau_m / Rin)
    'V0': -65.0,    # mV  initial voltage
}

FIT_PROTOCOLS = [2, 4, 6]         # VC trains, MC trains, simultaneous trains
DEFAULT_FILTER_LEN = 150          # ms
DEFAULT_HIST_LEN = 30             # ms
DEFAULT_VAL_FRAC = 0.20

# Exponential-Euler convention: which conductance advances V[t-1] -> V[t].
#   0 = conductance at the end of the step   (index t)    [default = original notebook]
#   1 = conductance at the start of the step (index t-1)
# This is a discretisation choice, not a bug: either is a valid scheme. Option 1
# shifts fitted filter latencies by one bin (1 ms). Default 0 reproduces the
# original notebook's likelihood exactly (checked numerically).
ODE_CONDUCTANCE_LAG = 0

# ── optimizer settings ───────────────────────────────────────────────────────
# A fit runs one or more L-BFGS-B "passes" in order; each pass starts from the
# best result so far. Each pass is (maxiter, ftol, gtol):
#   maxiter  max number of iterations in that pass
#   ftol     stop when one iteration improves the loss by less than this (relative)
#   gtol     stop when the largest gradient component is smaller than this
# A pass stops at whichever limit is hit first (r.message says which).
#
# Pick a preset by name anywhere a `passes` argument exists, e.g.
#   fit_all(..., passes='quick')   or   passes=[(500, 1e-6, 1e-5)]
PASS_PRESETS = {
    'quick': [(200, 1e-5, 1e-4)],                                        # smoke test
    'fast':  [(2000, 1e-5, 1e-4)],                                       # grid search
    'full':  [(2000, 1e-5, 1e-4), (3000, 1e-7, 1e-6), (5000, 1e-9, 1e-7)],
}
FULL_PASSES = PASS_PRESETS['full']
FAST_PASSES = PASS_PRESETS['fast']

# Max model evaluations per pass. scipy's default is 15000, but without an
# analytic gradient every iteration costs (n_params + 1) evaluations
# (~635 with fl=150, hl=30), so the default silently stops each pass after
# ~15-25 iterations, long before maxiter. None = no limit (maxiter decides).
MAXFUN = None