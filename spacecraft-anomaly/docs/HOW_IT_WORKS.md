# How It Works: Spacecraft Anomaly Detection

This document explains the project in detail: the problem, the data, every step of the method, every file in the code, how to run it, what it produces, how it was tested, and what it does not do yet. It is written so you can use it for the project report and the viva.

For setup and run commands only, see [`../README.md`](../README.md).

---

## Contents

1. [The problem](#1-the-problem)
2. [The data](#2-the-data)
3. [The big idea](#3-the-big-idea)
4. [The pipeline, step by step](#4-the-pipeline-step-by-step)
5. [Evaluation](#5-evaluation)
6. [Baselines and the ablation](#6-baselines-and-the-ablation)
7. [Code walkthrough, file by file](#7-code-walkthrough-file-by-file)
8. [Hyperparameters](#8-hyperparameters)
9. [Running experiments](#9-running-experiments)
10. [Outputs](#10-outputs)
11. [Testing and verification](#11-testing-and-verification)
12. [Differences from the paper](#12-differences-from-the-paper)
13. [Limitations and known issues](#13-limitations-and-known-issues)
14. [Possible extensions](#14-possible-extensions)
15. [Viva questions and answers](#15-viva-questions-and-answers)
16. [References](#16-references)

---

## 1. The problem

Spacecraft send back thousands of telemetry channels: temperatures, voltages, currents, wheel speeds, and so on. Engineers need to know quickly when something behaves strangely, such as a sensor drifting, a sudden spike, or a signal that has the right values but the wrong shape.

The hard parts are:

- **Labels are rare.** Very few anomalies are ever recorded, so a normal supervised classifier has almost nothing to learn from.
- **There are many channels.** Hand-tuned alarm limits for each channel do not scale, and they miss anomalies that stay inside the limits.
- **Context matters.** A value can be fine at one time and wrong at another. For example, a heater reading that is normal after a "heater on" command is abnormal without it.

This project learns what *normal* looks like for each channel and flags readings that do not match. It reimplements the method of Hundman et al. (KDD 2018) from NASA JPL, often called **Telemanom**.

---

## 2. The data

### Source

The data is real telemetry from two NASA missions, labeled by experts at NASA JPL using Incident Surprise Anomaly (ISA) reports:

| Mission | What it is | Labeled channels | Labeled anomalies | Features per timestep |
|---|---|---|---|---|
| **SMAP** | Soil Moisture Active Passive satellite | 54 | 69 | 25 |
| **MSL** | Mars Science Laboratory, the **Curiosity rover** | 27 | 36 | 55 |

The label counts were checked against the downloaded files: 105 anomalies in total.

### Download

`scripts/download_data.py` tries, in order:

1. The original S3 bucket `s3-us-west-2.amazonaws.com/telemanom/data.zip`. It currently returns **403 Access Denied**.
2. The public Kaggle mirror `patrickfleith/nasa-anomaly-detection-dataset-smap-msl`, a direct download of about 86 MB with no account needed. **This is the one that works.**
3. The same mirror through the `kaggle` CLI, if you have an API token.
4. A zip you downloaded yourself: `--zip path/to/archive.zip`.

The labels file always comes from the Telemanom GitHub repository.

### Files

```
data/
├── labeled_anomalies.csv
├── train/<chan_id>.npy   # e.g. train/P-1.npy, shape (2872, 25)
└── test/<chan_id>.npy    # e.g. test/P-1.npy,  shape (8505, 25)
```

- Each `.npy` file is one channel, stored as a 2-D array of shape `(timesteps, features)`.
- **Column 0** is the telemetry value. NASA already scaled it to the range [-1, 1], so the code does no extra scaling.
- **Columns 1 onwards** are one-hot encoded **commands** sent to the spacecraft. They let the model learn, for example, that a jump right after a command is expected.
- **Train** files are mostly nominal (normal). **Test** files contain the labeled anomalies.
- Channel IDs encode the subsystem: P = power, T = thermal, E, S, A, D, F, M, G, R, C, and so on. The exact meaning is not published.

### The labels file

```
chan_id,spacecraft,anomaly_sequences,class,num_values
P-1,SMAP,"[[2149, 2349], [4536, 4844], [3539, 3779]]","[contextual, contextual, contextual]",8505
S-1,SMAP,"[[5300, 5747]]",[point],7331
```

- `anomaly_sequences` are `[start, end]` index ranges (inclusive) into the **test** array.
- `class` says whether each anomaly is:
  - **point**: a value far outside the normal range, such as a spike. A simple alarm limit could catch it.
  - **contextual**: values that look normal on their own but are wrong for the situation, such as a drift or a wrong pattern. These need a model that understands time.

### Data quirks found and handled

| Quirk | How the code handles it |
|---|---|
| Channel **P-2** appears in two rows of the CSV | `load_labels()` merges the rows into one channel with both anomalies. This is why SMAP has 54 channels rather than 55. |
| Channel **T-10** has `.npy` files but no labels | It is skipped, because the channel list comes from the labels file. |
| The `class` column is not valid Python (`[point, contextual]` without quotes) | A custom parser handles it. |

---

## 3. The big idea

> **Predict, then compare.** Train a model to forecast each channel's next values using only normal data. When the forecast is badly wrong, the system is probably not behaving normally.

This is **unsupervised** with respect to anomalies:

- The model is trained on train data only, with no anomaly labels.
- The threshold is chosen from the errors themselves, again with no labels.
- The labels are used **only at the end, to measure** how well the detector did.

Why a forecaster instead of a classifier?

- With only 105 labeled anomalies across 81 channels, a classifier would overfit badly.
- A forecaster learns from every normal timestep, which gives hundreds of thousands of training examples.
- It also gives an **interpretable** signal: the plot shows exactly where prediction and reality split apart.

Why an LSTM?

- Telemetry has long-range temporal patterns, such as cycles, slow drifts and responses to commands. A Long Short-Term Memory network keeps a memory state across many steps, so it can model these patterns.
- It takes the command columns as extra inputs, so it can learn expected behavior after a command. This is key for *contextual* anomalies.

---

## 4. The pipeline, step by step

```
                    ┌───────────────────────────────────────┐
 train/<chan>.npy ─►│ 4.1 Windowing  (250 in → 10 out)       │
                    └───────────────┬───────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │ 4.2 Train LSTM (MSE, early stopping)   │──► models/<SC>/seed<N>/<chan>.pt
                    └───────────────┬───────────────────────┘
                                    ▼
 test/<chan>.npy ──►┌───────────────────────────────────────┐
                    │ 4.3 Forecast every test timestep       │
                    └───────────────┬───────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │ 4.4 Error = |actual − forecast|        │
                    │ 4.5 EWMA smoothing                     │
                    │ 4.6 Dynamic threshold ε (per window)   │
                    │ 4.7 Buffer + group into sequences      │
                    │ 4.8 Pruning                            │
                    └───────────────┬───────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
 labels ───────────►│ 5. Event-level evaluation              │──► results/<run>/
                    └───────────────────────────────────────┘
```

Everything below happens **separately for each channel**.

### 4.1 Windowing (`data.py`)

The time series is cut into overlapping sliding windows:

```
X[i] = series[i : i+250]              # 250 timesteps × all features (input)
y[i] = series[i+250 : i+260, col 0]   # next 10 telemetry values     (target)
```

- `l_s = 250` is the history length.
- `n_predictions = 10` is the forecast horizon.
- A train series of length T gives `T − 250 − 10 + 1` windows.

### 4.2 Training (`train.py`, `model.py`)

**Architecture** (`LSTMForecaster`):

```
input  (batch, 250, F)          F = 25 for SMAP, 55 for MSL
  → LSTM layer 1, 80 units
  → dropout 0.3
  → LSTM layer 2, 80 units
  → take the last hidden state (batch, 80)
  → dropout 0.3
  → Linear 80 → 10
output (batch, 10)              forecasts for the next 10 steps
```

That is 86,890 parameters for SMAP and 96,490 for MSL: small, fast, and one model per channel.

**Training procedure:**

- **Validation split:** the **last** 20% of training windows, in time order rather than shuffled, so the model is validated on the "future" of its training data.
- **Loss:** mean squared error between the forecast and the real next 10 values.
- **Optimizer:** Adam, learning rate 0.001, batch size 64, up to 35 epochs.
- **Early stopping:** stop when validation loss has not improved by at least 0.0003 for 10 epochs, then restore the best weights.
- **Reproducibility:** `set_seed(seed)` fixes Python, NumPy and PyTorch random seeds. Run seeds 0, 1 and 2 to report mean ± std.
- **Device:** CUDA GPU if available, then Apple MPS, then CPU.

The trained model is saved as a checkpoint containing the weights, config, number of features and training history. On the next run it is **loaded instead of retrained**, unless you pass `--retrain`.

### 4.3 Forecasting the test set (`train.predict_series`)

For the test series, every window `[i, i+250)` predicts timesteps `i+250 … i+259`. So each timestep `t ≥ 250` is predicted by up to 10 different windows, made 1 to 10 steps ahead. These predictions are **averaged** to give one forecast `ŷ[t]`.

The first 250 test timesteps have no forecast, since they are the first history window. All later results are shifted by 250 so they line up with the test indices in the labels.

### 4.4 Prediction error

```
e[t] = | y[t] − ŷ[t] |
```

### 4.5 Smoothing (`thresholding.smooth_errors`)

Raw errors are spiky, because a single noisy reading can produce one large error. They are smoothed with an **exponentially weighted moving average**:

```
e_s = EWMA(e, span = error_window × smoothing_perc) = EWMA(e, span = 2100 × 0.05 = 105)
```

Now a lasting rise in error stands out, and one-off blips fade.

### 4.6 Nonparametric dynamic threshold (`thresholding.find_epsilon`)

This is the paper's main contribution. Instead of assuming the errors are Gaussian, or hand-picking a limit, the threshold is chosen from the error distribution itself.

Candidate thresholds:

```
ε(z) = μ(e_s) + z · σ(e_s)      for z = 2.5, 3.0, 3.5, …, 11.5
```

For each candidate ε:

- `e_a` = the points with `e_s ≥ ε` (after adding a buffer, see 4.7)
- `E_seq` = the contiguous runs those points form
- `Δμ` = how much the **mean** of the errors drops when points above ε are removed
- `Δσ` = how much the **standard deviation** drops when they are removed

The score is:

```
            Δμ/μ  +  Δσ/σ
score(ε) = ─────────────────
            |e_a| + |E_seq|²
```

The chosen ε has the **highest score**.

What this means intuitively:

- **Numerator:** a good threshold cuts off the points that were inflating the mean and spread, which are the true outliers.
- **Denominator:** it is penalized for flagging many points, and *heavily* (squared) for flagging many separate regions. This favors a few clear anomalies over scattered noise.
- **Constraints:** a candidate is skipped if it flags 50% or more of the window, or more than 5 sequences.

**Windows.** The thresholding runs separately on consecutive chunks of `error_window = 2100` errors, so each part of a long mission gets its own threshold that adapts to local behavior. A short last chunk is merged into the one before it.

**Significance check.** If the largest error in a chunk is below 5% of the signal's range in that chunk, nothing is flagged there. This avoids false alarms on near-flat signals where tiny errors still cross a statistically chosen ε.

### 4.7 Buffer and grouping (`_buffered`, `group_sequences`)

Each flagged index is widened by `error_buffer = 100` steps on both sides. The flagged indices are then grouped into contiguous **sequences** `(start, end)`. Nearby flags become one event instead of many tiny ones, which matches how anomalies are labeled: as ranges.

### 4.8 Pruning (`thresholding.prune`)

Even with a good ε, some flagged sequences are only slightly above normal. Pruning removes them:

1. For every flagged sequence, take its **maximum** smoothed error.
2. Sort these maxima in descending order, and append the maximum error of the **non-flagged** points at the end.
3. Compute the percentage drop between each pair of neighbors: `d_i = (max_i − max_{i+1}) / max_i`.
4. Find the **last** position where `d_i > p` (`p = 0.13`, i.e. a 13% drop).
5. Keep the sequences up to that position. Everything after it is reclassified as normal.

Example, with sorted maxima `[1.00, 0.95, 0.40, 0.38]` and a normal maximum of `0.36`:

```
drops:   1.00→0.95 = 5%    0.95→0.40 = 58%    0.40→0.38 = 5%    0.38→0.36 = 5%
last drop > 13% is at position 2  →  keep the top 2 sequences, prune the other 2
```

The idea: real anomalies stand clearly above everything else, while borderline sequences sit close to the noise level.

The final flagged sequences are shifted by 250 back to test-array indices and passed to evaluation.

---

## 5. Evaluation

`evaluate.py` uses **event-level** scoring, as in the paper.

| Outcome | Rule |
|---|---|
| **True positive (TP)** | A labeled anomaly overlaps **at least one** flagged sequence |
| **False negative (FN)** | A labeled anomaly overlaps **no** flagged sequence (missed) |
| **False positive (FP)** | A flagged sequence overlaps **no** labeled anomaly (false alarm) |

Counts are summed over all channels of a spacecraft, then:

```
precision = TP / (TP + FP)          "of the alarms raised, how many were real?"
recall    = TP / (TP + FN)          "of the real anomalies, how many were caught?"
F1        = 2PR / (P + R)
F0.5      = 1.25 PR / (0.25 P + R)  weights precision more (fewer false alarms for operators)
```

It also reports **recall by class**: what fraction of *point* anomalies and of *contextual* anomalies were caught. Contextual anomalies are where an LSTM should beat simple statistics.

**Why event-level and not "point-adjust"?** Many papers use point-adjust: if any single point of an anomaly is detected, *every* point of it counts as detected, and then point-wise F1 is computed. Wu & Keogh (2021) and others showed this inflates scores so much that even random detectors can look strong. Event-level counting is stricter and easier to interpret.

---

## 6. Baselines and the ablation

A model is only impressive compared with something. Four methods are implemented (`--method`):

| Method | Score per timestep | Threshold | Purpose |
|---|---|---|---|
| `lstm` | Smoothed LSTM forecast error | Dynamic ε with pruning | **Main model** |
| `lstm_static` | Same smoothed LSTM error | Fixed: mean + 3σ of the test errors | **Ablation:** does the dynamic threshold help? |
| `zscore` | `|x_t − rolling mean| / rolling std` over the previous 250 steps (column 0) | 99.9th percentile of the scores on **train** | Simple statistical baseline |
| `iforest` | Isolation Forest anomaly score on features of the last 30 steps (values, differences, commands) | 99.9th percentile of the scores on **train** | Classic ML baseline |

Notes:

- The baseline thresholds come from the **train** data only, so no test labels leak in.
- `lstm_static` reuses the saved LSTM models, so it costs no extra training.
- All methods use the same buffer (100 steps), grouping and event-level evaluation, so the comparison is fair.
- The rolling z-score is expected to be weak on strongly periodic signals. That is a fair finding to discuss, not a bug.

---

## 7. Code walkthrough, file by file

```
spacecraft-anomaly/
├── README.md                  setup and run guide
├── docs/HOW_IT_WORKS.md       this document
├── pyproject.toml             package definition (pip install -e .)
├── requirements.txt           plain dependency list
├── .gitignore                 ignores data/, models/, results/
├── scripts/
│   └── download_data.py
├── src/sad/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── thresholding.py
│   ├── evaluate.py
│   ├── baselines.py
│   ├── run.py
│   ├── summarize.py
│   └── plot.py
└── tests/
    ├── conftest.py
    ├── test_data.py
    ├── test_thresholding.py
    ├── test_evaluate.py
    └── test_end_to_end.py
```

`sad` stands for **S**pacecraft **A**nomaly **D**etection.

### `scripts/download_data.py`
| Function | What it does |
|---|---|
| `fetch(url, dest)` | Downloads a URL to a file and returns False on failure |
| `install_from_zip(zip, out)` | Extracts only `.npy` and `.csv` files, finds the `train/` + `test/` pair anywhere inside (the Kaggle zip nests them under `data/data/`), and copies them to `data/` |
| `find_split_dirs(root)` | Searches for a `train/` folder with a sibling `test/` folder, both containing `.npy` files |
| `try_kaggle(tmp)` | Uses the `kaggle` CLI if installed |
| `main()` | Tries S3, then the direct Kaggle URL, then the Kaggle CLI, then downloads the labels and prints a summary |

### `src/sad/config.py`
A single `Config` dataclass holding every hyperparameter (see [§8](#8-hyperparameters)). `run.py` turns each field into a command-line flag automatically.

### `src/sad/data.py`
| Function | What it does |
|---|---|
| `data_dir()` | Resolves the data folder: argument, else `$SAD_DATA_DIR`, else `./data` |
| `load_labels()` | Reads the CSV, parses the sequences and classes, and merges the duplicate P-2 rows |
| `channels(spacecraft)` | Lists the channel IDs for SMAP or MSL |
| `load_channel(chan)` | Returns the `(train, test)` arrays as float32 |
| `make_windows(arr, l_s, h)` | Training windows with targets (vectorized with NumPy indexing) |
| `make_input_windows(arr, l_s)` | Test-time windows, no targets |
| `sequences_to_mask()` | Converts `[start, end]` ranges into a boolean array |

### `src/sad/model.py`
`LSTMForecaster(n_features, hidden_size, num_layers, dropout, n_predictions)` is a PyTorch `nn.Module` with an LSTM stack, dropout and a linear head.

### `src/sad/train.py`
| Function | What it does |
|---|---|
| `set_seed(seed)` | Makes runs reproducible |
| `default_device()` | Picks CUDA, then MPS, then CPU |
| `train_channel(train, cfg, seed)` | Builds windows, does the chronological train/validation split, runs the training loop with early stopping, and returns the best model and its history |
| `predict_series(model, series, cfg)` | Forecasts every timestep after the first `l_s`, averaging over the horizon |
| `save_model()` / `load_model()` | Saves and loads checkpoints (`.pt`) |

### `src/sad/thresholding.py`
| Function | What it does |
|---|---|
| `smooth_errors(y, ŷ, cfg)` | Computes absolute error, then EWMA |
| `find_epsilon(e_s, cfg)` | The dynamic-threshold search from [§4.6](#46-nonparametric-dynamic-threshold-thresholdingfind_epsilon) |
| `group_sequences(idx)` | Turns flagged indices into `(start, end)` runs |
| `_buffered(idx, buffer, n)` | Widens each flagged index by ±buffer |
| `prune(e_s, seqs, p)` | Pruning from [§4.8](#48-pruning-thresholdingprune) |
| `detect(e_s, cfg, y_true)` | Runs the whole thing per error window and returns the flagged sequences and the threshold at each timestep |
| `static_threshold(scores, t, buffer)` | Applies a fixed threshold, used by the ablation and baselines |

### `src/sad/evaluate.py`
| Function | What it does |
|---|---|
| `evaluate_channel(true, pred, classes)` | TP / FP / FN for one channel, plus hits by class |
| `prf(tp, fp, fn, beta)` | Precision, recall and F-beta |
| `aggregate(rows)` | Sums over channels and computes overall metrics and recall by class |

### `src/sad/baselines.py`
| Function | What it does |
|---|---|
| `zscore_scores(train, test)` | Rolling z-score of column 0; threshold from train |
| `iforest_scores(train, test)` | Isolation Forest (200 trees) on windowed features; trains on up to 20,000 subsampled train windows; threshold from train |

### `src/sad/run.py` (the main command)
1. Parses the arguments and builds a `Config`.
2. Loads the labels and picks the channels: all of them, or `--channels`.
3. For each channel:
   - **LSTM methods:** loads or trains the model, forecasts, smooths the errors, thresholds (dynamic or static), and shifts the indices by `l_s`.
   - **Baselines:** computes the scores and threshold, then groups into sequences.
   - Evaluates against the labels, saves a `.npz` file with the signal, forecast, score and threshold, and prints a progress line.
4. Aggregates, then writes `per_channel.csv` and `summary.json` and prints the metrics.
5. Optionally logs everything to MLflow (`--mlflow`).

### `src/sad/summarize.py`
Reads several `summary.json` files, groups them by (method, spacecraft), and prints **mean ± std** of precision, recall, F1 and F0.5 with the number of seeds. This is ready for the report table.

### `src/sad/plot.py`
Makes a two-panel figure for each channel:
- **Top:** the real telemetry (black) and the LSTM forecast (orange).
- **Bottom:** the anomaly score (purple) and the threshold (dashed).
- **Shading:** red is the labeled anomaly and blue is what was detected. The title shows TP / FP / FN.

### `tests/`
See [§11](#11-testing-and-verification).

---

## 8. Hyperparameters

All of these live in `src/sad/config.py`. The defaults follow the Telemanom paper and its reference config, and every one can be overridden on the command line, e.g. `--l-s 200`.

| Name | Default | Meaning |
|---|---|---|
| `l_s` | 250 | Input history length (timesteps) |
| `n_predictions` | 10 | Forecast horizon |
| `hidden_size` | 80 | LSTM units per layer |
| `num_layers` | 2 | LSTM layers |
| `dropout` | 0.3 | Dropout rate |
| `epochs` | 35 | Maximum training epochs |
| `batch_size` | 64 | Training batch size |
| `lr` | 0.001 | Adam learning rate |
| `validation_split` | 0.2 | Last fraction of train windows used for validation |
| `patience` | 10 | Early-stopping patience (epochs) |
| `min_delta` | 0.0003 | Minimum validation improvement that counts |
| `smoothing_perc` | 0.05 | EWMA span = `error_window × smoothing_perc` |
| `error_window` | 2100 | Length of each error chunk thresholded separately |
| `error_buffer` | 100 | Steps added around each flagged point |
| `p` | 0.13 | Minimum percentage drop for pruning |
| `z_range` | (2.5, 12, 0.5) | z values tried for ε (not a command-line flag) |

Good candidates to study in the report: `l_s`, `error_buffer`, `p`, and `smoothing_perc`. Change one at a time and use a different `--run-name` for each run.

---

## 9. Running experiments

Install and download first (see the README):

```bash
cd spacecraft-anomaly
```

```bash
pip install -e ".[dev]"
```

```bash
python scripts/download_data.py
```

**Quick check** (2 channels, few epochs):

```bash
python -m sad.run --spacecraft SMAP --channels P-1 E-1 --epochs 5 --run-name smoke
```

**Full experiment plan** for the report, best done on a Colab or Kaggle GPU.

1. The main model, 3 seeds × 2 spacecraft:
   ```bash
   for s in 0 1 2; do for sc in SMAP MSL; do python -m sad.run --spacecraft $sc --method lstm --seed $s; done; done
   ```
2. The ablation and baselines. These reuse the trained LSTMs, so they are fast:
   ```bash
   for s in 0 1 2; do for sc in SMAP MSL; do for m in lstm_static zscore iforest; do python -m sad.run --spacecraft $sc --method $m --seed $s; done; done; done
   ```
3. The results table:
   ```bash
   python -m sad.summarize results/*_seed*/summary.json
   ```
4. Plots for the report, for example a caught anomaly, a missed one, and a false alarm:
   ```bash
   python -m sad.plot results/lstm_SMAP_seed0
   ```

**Hyperparameter study**, for example the effect of the pruning parameter. Because models are reused, only the thresholding re-runs:

```bash
for p in 0.05 0.10 0.13 0.20; do python -m sad.run --spacecraft SMAP --method lstm --seed 0 --p $p --run-name lstm_SMAP_p$p; done
```

Note that `summarize` groups by method and spacecraft, so summarize these sweep runs separately from the seed runs.

**MLflow tracking:** add `--mlflow` to any run, then view the runs with:

```bash
mlflow ui
```

---

## 10. Outputs

```
models/
└── SMAP/seed0/P-1.pt              trained weights + config + training history

results/lstm_SMAP_seed0/
├── summary.json                   overall metrics, recall by class, full config
├── per_channel.csv                one row per channel
├── channels/P-1.npz               arrays for plotting
└── plots/P-1.png                  created by sad.plot
```

**`summary.json`** fields: `run_name, method, spacecraft, seed, n_channels, config, tp, fp, fn, precision, recall, f1, f0.5, recall_by_class`.

**`per_channel.csv`** columns: `chan_id, tp, fn, fp, n_pred, pred_seqs, true_seqs`, plus `epochs_run, best_val_loss` when the model was trained in that run. Use it to find which channels fail and to choose examples for error analysis.

**`channels/<chan>.npz`** arrays: `signal` (test column 0), `scores` (smoothed error or baseline score), `threshold`, and `y_hat` (LSTM forecast, LSTM methods only). The first `l_s` entries are NaN for LSTM methods.

---

## 11. Testing and verification

### Automated tests

Run them with `pytest`, or `python -m pytest`. They use **synthetic** telemetry, so no download is needed: a sine wave with noise and command columns, and an injected shift in the test split.

| Test file | What it checks |
|---|---|
| `test_data.py` | Window shapes and targets are correct; test windows cover all future steps; label masks are inclusive; the duplicate-channel merge works |
| `test_thresholding.py` | Sequence grouping; the dynamic threshold separates an injected error spike from noise and `detect` returns exactly one sequence covering it; pruning drops a sequence barely above the noise |
| `test_evaluate.py` | TP / FP / FN counting with overlaps and recall by class; aggregate precision, recall and F-scores |
| `test_end_to_end.py` | The full LSTM pipeline trains, saves models, writes results, **reuses saved models on a second run**, and plots; Isolation Forest finds the injected anomaly; the z-score baseline runs |

Current status: **11 tests pass.**

### Manual checks done

| Check | Result |
|---|---|
| Full LSTM run on 2 synthetic channels (15 epochs) | Both injected anomalies caught, 0 false alarms; the plot showed the forecast tracking the sine wave and the error spiking only at the anomaly |
| Real dataset downloaded through the script | 82 train + 82 test files, 81 labeled channels, 105 anomalies |
| Real file shapes | P-1: train (2872, 25), test (8505, 25); M-1: train (2209, 55), test (2277, 55) |
| Pipeline runs on real SMAP (P-1) and MSL (M-1) channels | Runs without errors |
| `--zip` install from a nested archive | Works |

**Not done yet:** full training on the real data. The 3-epoch CPU smoke test above was only to check that the code runs; its detections mean nothing. **No real results exist yet.** Every number in your report must come from your own logged runs.

---

## 12. Differences from the paper

| Aspect | Paper / reference code | This project |
|---|---|---|
| Framework | Keras / TensorFlow | PyTorch |
| Error windows | Overlapping windows with a lookback buffer | Consecutive non-overlapping chunks of 2,100 errors; a short tail is merged into the previous chunk |
| Low-error side | Also thresholds unusually *low* errors (inverted errors) | Only high errors |
| Horizon aggregation | Several options | Mean of all forecasts for a timestep |
| Validation split | Keras `validation_split` (last 20%) | Same idea: the last 20% of windows, in time order |
| Baselines | Compared mainly with other thresholding variants | Adds rolling z-score, Isolation Forest and a static-threshold ablation |

Expect results that are **similar in spirit but not identical** to the paper. Report your own numbers.

---

## 13. Limitations and known issues

- **One model per channel.** Cross-channel relationships, such as one sensor affecting another, are not used.
- **Only column 0 is forecast.** The commands are inputs only.
- **The static ablation uses test statistics** (mean + 3σ of test errors). It is still label-free, but not strictly "train-only". Mention this when you discuss it.
- **Benchmark caveats.** Wu & Keogh (2021) argue that some anomalies in this dataset are trivial, that some labels may be imprecise, and that anomalies tend to sit near the end of test sets. Because of this, strong baselines matter, and conclusions should be modest.
- **Hyperparameters are not tuned on labels.** This is deliberate, to stay unsupervised. Tuning `p` or `error_buffer` on the *test* labels would be cheating. If you tune, use a held-out set of channels.
- **Training time.** 81 models with up to 35 epochs each takes a while on CPU. Use a GPU (Colab or Kaggle).

---

## 14. Possible extensions

- An **LSTM autoencoder** (reconstruction error) as an alternative to forecasting.
- **GRU, TCN or a small Transformer** forecasters, as a model ladder.
- **Low-error thresholding**, as in the paper, to catch signals that become unusually flat.
- **Multivariate models** across channels of the same subsystem.
- **Tuning on held-out channels**, for example tuning `p` on half of the channels and testing on the other half.
- A **dashboard** (Streamlit or React) to browse channels and their detections.
- **Latency analysis:** how many timesteps after an anomaly starts is it first flagged?

---

## 15. Viva questions and answers

**Q: Why train only on normal data?**
Anomalies are rare and varied, so there are too few to learn from. By learning normal behavior, the model can flag *any* deviation, including types it has never seen.

**Q: Why forecast instead of classify?**
A classifier needs many labeled anomalies, and there are only 105. A forecaster uses every normal timestep as training data, and its error is an interpretable anomaly score.

**Q: Why an LSTM and not a plain feed-forward network?**
Telemetry depends on long histories and on earlier commands. An LSTM's gated memory keeps that context over hundreds of steps.

**Q: Why include the command columns?**
They explain *expected* changes. A jump after a command is normal; the same jump without one is not. This is what makes contextual anomalies detectable.

**Q: Why smooth the errors?**
Single noisy points produce one-off large errors. EWMA smoothing keeps lasting deviations and suppresses blips, which reduces false alarms.

**Q: Explain the dynamic threshold.**
Try ε = μ + zσ for several z. Pick the one where removing the errors above ε lowers the mean and standard deviation the most, penalized by how many points and sequences get flagged. It needs no labels and no Gaussian assumption, and it is recomputed per window, so it adapts over time.

**Q: Why not just use mean + 3σ?**
Errors are not Gaussian and differ hugely between channels. A fixed rule either floods noisy channels with alarms or misses subtle anomalies. The `lstm_static` ablation measures this directly.

**Q: What does pruning do?**
It removes flagged sequences whose peak error is not clearly above the next one. Real anomalies stand out sharply; borderline ones are usually noise.

**Q: How do you evaluate, and why that way?**
Event-level: a labeled anomaly is caught if any flagged sequence overlaps it, and a flag with no overlap is a false positive. Point-adjusted F1 is known to inflate scores, so it is avoided.

**Q: What is the difference between point and contextual anomalies?**
Point anomalies are values outside the normal range, which a simple limit can catch. Contextual anomalies are normal-looking values that are wrong for the moment, such as a drift or a wrong pattern. These need temporal modeling, so recall by class shows where the LSTM adds value.

**Q: Why F0.5 as well as F1?**
For operators, false alarms are costly ("alarm fatigue"). F0.5 weights precision more heavily.

**Q: How do you make sure there is no data leakage?**
Models train on train data only. Validation is the last 20% of train windows in time order. Thresholds come from errors (LSTM) or train scores (baselines), never from labels. Labels are used only in `evaluate.py`.

**Q: Why run 3 seeds?**
Neural network training is random, through initialization and batch order. Reporting mean ± std over 3 seeds shows whether differences between methods are real or just noise.

**Q: What are the weaknesses of this dataset?**
Wu & Keogh (2021) show that some anomalies are trivially easy, some labels may be imprecise, and anomalies are often near the end of test sets. That is why simple baselines are included and claims are kept modest.

**Q: What would you do with more time?**
See [§14](#14-possible-extensions): autoencoders and Transformers, multivariate models, low-error thresholding, tuning on held-out channels, and a dashboard.

---

## 16. References

1. K. Hundman, V. Constantinou, C. Laporte, I. Colwell, T. Soderstrom. **Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.** KDD 2018. https://arxiv.org/abs/1802.04431
2. Reference implementation and labels: https://github.com/khundman/telemanom
3. Dataset mirror: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
4. R. Wu, E. Keogh. **Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress.** IEEE TKDE, 2021.
5. S. Hochreiter, J. Schmidhuber. **Long Short-Term Memory.** Neural Computation, 1997.
6. F. T. Liu, K. M. Ting, Z.-H. Zhou. **Isolation Forest.** ICDM 2008.
