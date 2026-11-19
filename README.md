# CBEM — Conductance-Based Encoding Model for MC / VC optogenetic stimulation

Fits, for each recorded neuron, how optogenetic stimulation of **motor cortex (MC)** and
**visual cortex (VC)** drives **excitatory (ge)** and **inhibitory (gi)** conductances, and
how those conductances produce spikes. It then compares MC-only, VC-only and simultaneous
stimulation to ask whether the two inputs sum linearly, sub-linearly (depletion) or
super-linearly (facilitation).

Run everything from `run_cbem.ipynb`. The notebook is thin: all logic lives in the `cbem/` package.

---

## 1. Quick start

```bash
pip install -r requirements.txt
jupyter notebook run_cbem.ipynb      # start it from THIS folder so `import cbem` works
```

In the notebook, edit the **Settings** cell (`MAT_FILE`, `MAT_VARIABLE`, `FILTER_LEN`, ...)
and run top to bottom. Set `RUN_GRID_SEARCH = True` to pick hyper-parameters first
(see section 5). Fitted parameters are cached in `outputs/cbem_results.pkl`; set
`REFIT = False` to reload them instead of refitting.

| File | Contents |
|---|---|
| `run_cbem.ipynb` | Thin driver notebook: settings + one call per step |
| `cbem/config.py` | Constants, protocol names, BIOPHYS defaults, optimiser passes |
| `cbem/data.py` | `.mat` loading, 30 kHz → 1 kHz, MC/VC split, fit set, design matrix |
| `cbem/model.py` | Conductances → voltage ODE → spike probability, likelihood |
| `cbem/fitting.py` | L-BFGS-B fitting, cross-validation, grid search, save/load |
| `cbem/analysis.py` | Filter latencies/widths, E/I, decomposition, PSTH, simultaneous, MI, summation |
| `cbem/plotting.py` | Every figure (pass `outdir=` to also save PDFs) |

---

## 2. Data format

`.mat` variable of shape `(N_samples at 30 kHz, 5)`:

| Column | Content |
|---|---|
| A | protocol ID (1–6) — **not** a time column |
| B | merged laser channel (0/1) |
| C, D, E | spike trains of neuron 1, 2, 3 (0/1) |

| ID | Protocol | MC laser | VC laser |
|---|---|---|---|
| 1 | VC single pulses | – | on |
| 2 | VC 100 Hz trains | – | on |
| 3 | MC single pulses | on | – |
| 4 | MC 100 Hz trains | on | – |
| 5 | Simultaneous single pulses | on | on |
| 6 | Simultaneous 100 Hz trains | on | on |

The raw data has ONE merged laser column. `data.prepare_dataset` splits it into an MC and a
VC channel from the protocol ID (protocols 5/6 put the same pulse on both channels).

---

## 3. How the pipeline works (main points)

1. **Downsample 30 kHz → 1 kHz.** Binary signals use the *max* over each 30-sample window, so
   no spike or pulse is lost. Protocol ID takes the first sample of each window.
2. **Design matrix (time-lagged / Toeplitz).** For each of the 2 laser channels, 150 columns
   hold the stimulus at lag 0, 1, … 149 ms. Shape `(T, 300)`. Learning a weight per column =
   learning a temporal **filter**: how much a pulse *k* ms ago contributes now.
3. **Conductances.** `ge = softplus(X·k_e + dc_e)`, `gi = softplus(X·k_i + dc_i)`.
   `softplus` keeps conductances positive. Each has an MC filter and a VC filter.
4. **Voltage.** Leaky membrane with excitatory and inhibitory conductances, solved by
   exponential Euler: V decays toward `V_inf = (gl·El + ge·Ee + gi·Ei) / (gl+ge+gi)` with
   factor `exp(-dt·g_tot/Cm)`.
5. **Spike probability.** `lam = softplus(a·(V − Vth) + h·spike_history) · dt`.
   `h` is a 30 ms post-spike filter (refractoriness / adaptation).
6. **Likelihood.** Bernoulli: `−Σ [y·log λ + (1−y)·log(1−λ)]`.
7. **Fitting.** Three successive L-BFGS-B passes with tightening tolerances, each starting
   from the previous best. Parameters per neuron: `4·filter_len + hist_len + 4`.
8. **Read-outs.** Filters (shape, peak latency, width), **E/I ratio** (integrated positive
   area of excitatory ÷ inhibitory filter per pathway), conductance traces, PSTH vs model,
   voltage-vs-history decomposition.

### Biophysical constants — update from your patch data

`cbem/config.py → BIOPHYS` ships with placeholders (El −65 mV, Ee 0, Ei −55, gl 0.9 nS,
Cm 0.99 nF). Replace with your measurements:

```python
gl = 1000 / R_in_MOhm            # nS
Cm = (tau_m_ms / 1000) / (R_in_MOhm / 1000)   # nF
El = V0 = V_rest                 # mV
```

---

## 4. Simultaneous stimulation: what the analysis does and its limits

In protocols 5/6 the same pulse drives **both** channels, so MC and VC filters are active
together and spiking is a nonlinear function of their sum.

* **Cross-constraint caveat.** Fitting on simultaneous data forces the MC and VC filters to
  jointly explain it. If the real interaction is strongly nonlinear (gain change, depletion,
  saturation), the model can misattribute it — e.g. enhanced simultaneous excitation can
  inflate both excitatory filters and distort E/I ratios.
* **Interaction conductance.** Run the model on MC-only, VC-only and simultaneous stimulus;
  compare `ge_sim`, `gi_sim` with the linear sum. The difference shows *where* the
  nonlinearity lives (excitation, inhibition, or both).
* **Spike-count modulation index:** `MI = (r_sim − (r_mc + r_vc)) / (r_sim + r_mc + r_vc)`.
  0 = linear, >0 super-additive, <0 sub-additive / depletion. Needs no model.
* **Predicted vs actual rate:** model-based linear prediction `lam_mc + lam_vc` vs the
  predicted and observed simultaneous rate.
* **Conductance-level summation:** `Δg_sim = α·Δg_mc + β·Δg_vc` on baseline-removed
  conductances. α = β = 1 linear; <1 depletion / shared resources; >1 facilitation.
* **α / β identifiability.** In protocol 5 the same pulse drives both channels, so the two
  regressors differ only through the filter shapes. If the MC and VC filters are similar, α
  and β are strongly collinear and can take wild opposite-signed values while the fit (R²)
  stays good. Trust α+β and R²; treat the individual values with caution.
* **Remaining limitation.** All predictions use the *same fitted filters*, so nonlinearity
  already absorbed into the filters during fitting does not show as a residual. The cleanest
  test is to fit on single-pathway protocols only (`FIT_PROTOCOLS = [2, 4]`) and predict the
  simultaneous protocol without refitting.
* **Spike-level vs conductance-level.** Even with perfectly linear conductances, softplus,
  voltage dynamics and the threshold can make spike counts sublinear. Report both.
* **Open idea (not implemented).** An explicit MC×VC interaction channel
  (`X_int = X_mc * X_vc`, own e/i filters, `6·fl + hl + 4` parameters) would let the model
  represent a real conductance-level interaction. Compare cross-validated likelihood with
  and without it.

---

## 5. Choosing hyper-parameters (grid search)

`fitting.grid_search` sweeps `filter_len × hist_len × lambda_l2` on a short,
**protocol-balanced** subset (first ~60 s split evenly across fitted protocols), holds out the
tail of every segment, and ranks candidates by the **worst** validation loss per spike across
neurons — so the winner is acceptable for all of them, not just the best-behaved one.

* The L2 penalty acts on the stimulus filters only (not on `h`, `dc`, `a`, `Vth`).
* **λ is rescaled** (`scale_lambda`) before the full fit: the likelihood grows with data but
  the penalty does not, so a λ tuned on a short subset would otherwise be relatively too
  strong on the full dataset.
* Use ≥120 s if any neuron fires below ~2 Hz. Neurons with <5 validation spikes are skipped.

---

## 6. Cross-validation

`fitting.cross_validate` refits on the first 80 % of **every** segment and scores the rest.
Losses are in nats per spike for train, validation and a constant-rate null model, so they are
directly comparable. Validation ≫ train ⇒ overfitting (shorten `filter_len` or raise
`lambda_l2`). Validation ≈ null ⇒ the model learned nothing beyond the mean rate.

---

## 7. Conventions worth knowing

* **Segments.** Every contiguous same-protocol run is a segment. Lags, voltage and spike
  history never cross a segment boundary; the voltage restarts at `V0` in each segment.
* **ODE timing** (`config.ODE_CONDUCTANCE_LAG`): 0 (default) advances V[t−1]→V[t] with the
  conductance at t, exactly as in the original notebook (likelihood checked to match to
  machine precision); 1 uses the conductance at t−1. Both are valid discretisations, not a
  bug; option 1 shifts fitted filter latencies by 1 ms.
* **Spike history indexing** was already causal (strictly past spikes) in the original
  notebook; the rewritten version gives identical values.
* **PSTH onsets.** For trains only the first pulse of each train is an onset
  (`min_gap_ms = 50`). The model is simulated over the whole run, then cropped, so the
  voltage carries its true history.
* **Peak latency/width.** Dominant lobe = largest |value|; a negative lobe is analysed
  sign-flipped; width is FWHM.

---

## 8. What changed vs. the earlier notebooks

**Real bugs fixed**
* `spk1_1k / spk2_1k / spk3_1k` were never defined (NameError) → all spike access goes
  through the dataset (`ds.spk_1k[i]`).
* The summation test counted the baseline `dc_e / dc_i` twice → conductance changes are now
  baseline-removed before regressing (and the linear prediction counts the baseline once).
  (The first rewrite only fixed the plotted prediction, not the regression itself.)
* Misleading "col A: time" comment (col A is the protocol ID).
* `make_p0` was called before it was defined.

**Previously reported as bugs, but were not**
* Spike-history indexing and ODE look-ahead: the original code was already causal; my
  earlier "fixes" were equivalent / a convention change. See section 7.

**Behaviour changes (methodological fixes)**
* **Segments:** voltage, history and stimulus lags no longer run across the jump between
  different protocol runs.
* **Validation split:** the tail of *every* segment is held out. A single tail split of the
  concatenated data would have been mostly the last protocol.
* **Cross-validation** now uses the same L2 as the main fit (it silently used none before),
  and plots per-spike losses with a constant-rate null (train and validation totals were
  compared on different amounts of data before).
* **Grid search** uses a protocol-balanced subset (the first 60 s of the concatenated data
  could have been a single protocol) and rescales lambda to the full data size.
* **PSTH:** model simulated over the whole run then cropped (windows used to restart the
  voltage and ignore earlier pulses); trains aligned to the first pulse.
* **Filter statistics:** dominant lobe = largest |value| (a larger negative lobe used to be
  ignored).
* No reliance on notebook globals (`best_lambda`, `fl`, `hl`, ...): everything is passed
  explicitly.

---

## 9. If the fit looks bad

1. **Sanity plot:** spikes should appear a few ms after laser pulses.
2. **Tiny loss drop / flat filters:** the neuron may not respond to the laser — check the
   raster first, or try `filter_len = 50`.
3. **MI very negative:** check protocols 1 and 3 actually drive the cell on their own.
4. **α or β ≫ 1:** possible filter overfitting — raise `lambda_l2`.
5. **Validation ≫ train:** reduce `filter_len` or raise `lambda_l2`.
6. **PSTH mismatch:** check `PRE_MS` / `POST_MS` bracket the response window.
7. **Slow fitting:** install numba. The optimiser uses finite-difference gradients (one model
   evaluation per parameter per step), which dominates run time; an autodiff gradient
   (JAX / PyTorch) would be the next big speed-up.
