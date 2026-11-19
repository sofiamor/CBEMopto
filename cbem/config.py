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

# (maxiter, ftol, gtol) for successive L-BFGS-B passes
FULL_PASSES = [(2000, 1e-5, 1e-4), (3000, 1e-7, 1e-6), (5000, 1e-9, 1e-7)]
FAST_PASSES = [(2000, 1e-5, 1e-4)]
