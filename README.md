# energy-reso-fitter

Fitting tools to extract the energy resolution of the CMS ECAL from H4 test-beam data (2026 test beam campaign, run at 340/400/500 Ω preamp feedback resistance).

Input histograms (uncalibrated 3x3 amplitude spectra per beam energy) are produced by a separate DQM/analysis repo and are read directly as ROOT files by the scripts here.

- Histograms: https://rgargiul.web.cern.ch/reso-energy-ecal-h4-dqm-2026/
- Repo that creates the histograms: `git@github.com:raeubaen/prompt-h4-cms-ecal-analysis.git`

## What it does

1. For each beam energy, fit the uncalibrated 3x3 amplitude histogram with a double-sided Crystal Ball function to extract the peak position and resolution (`fit.sh` + `dcb.cxx`).
2. Collect the per-energy fit results into a CSV (`en,sigma_abs,zero,err_sigma_abs,peak_abs,err_peak_abs`).
3. Fit the resulting `σ/E` vs. `E` points with the standard calorimeter resolution parametrization and produce resolution plots (`fit_plot.sh`).

## Requirements

- ROOT (with `thisroot.sh` sourced, e.g. from `~/root_build/bin/thisroot.sh`)

## Usage

### 1. Per-energy peak/resolution fit

```sh
./fit.sh <energy_list.txt> <output.csv> <input_dir> <resistance>
```

- `energy_list.txt`: newline-separated list of beam energies in GeV (see `energies.txt` for an example).
- `output.csv`: path for the resulting fit-results CSV.
- `input_dir`: directory containing `FitAmp_3x3_<energy>.root` histogram files.
- `resistance`: preamp feedback resistance, one of `340`, `400`, `500` (sets the ADC-to-energy scale used to pick the fit range).

For each energy, the script opens `FitAmp_3x3_<energy>.root`, restricts the fit range around the expected peak, runs the double-sided Crystal Ball fit (`dcb`) iteratively, appends the fit parameters to the output CSV, and saves the fitted histogram as `FitAmp_3x3_<energy>_fitted.root` in `input_dir`.

### 2. Resolution curve fit

```sh
./fit_plot.sh <reso.csv> <resistance>
```

Reads the CSV produced above, builds a `TGraphErrors` of `σ/E` vs. beam energy, and fits it with:

```
σ/E = sqrt( (N/E)^2 + (S/√E)^2 + C^2 )
```

Outputs (named with the given `resistance`):
- `reso_interactive_<resistance>.root` — graph, fit function, and canvas (interactive)
- `reso_canvas_<resistance>.root` — saved canvas

## Repository layout

- `dcb.cxx` — double-sided Crystal Ball function and `dcb()` fit helper (loaded into ROOT via `.L`).
- `fit.sh` — per-energy histogram fitting driver.
- `fit_plot.sh` — resolution-vs-energy fit and plotting driver.
- `energies.txt` — example energy list for the 340 Ω dataset.
- `prompt_*ohm*/` — per-resistance working directories with raw and fitted ROOT histograms.
- `reso_*.csv`, `reso_*.root` — fit results and output plots per resistance setting.

Bulk data files (`*.root`, `*.csv`, `prompt_*` directories) are gitignored; only scripts and small reference files are tracked.
