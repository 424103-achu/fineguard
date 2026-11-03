# Spacecraft Anomaly Detection

An LSTM learns how spacecraft sensor readings normally behave and flags readings that look unusual, such as a drifting sensor or a sudden spike. The data is real telemetry from NASA's **SMAP** satellite and the **Curiosity rover (MSL)**, with anomalies labeled by experts at NASA JPL.

This is a reimplementation of Hundman et al., *[Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://arxiv.org/abs/1802.04431)*, KDD 2018 (the "Telemanom" method).

> **Full explanation of the method, code, experiments and viva Q&A: [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md).**

> This folder is a standalone project. It shares nothing with the Fine-Print Guard code in the rest of the repository.

## How it works

```
train/<chan>.npy (nominal) ──► LSTM forecaster (one per channel)
                                       │
test/<chan>.npy ──► forecast next values ──► |actual − forecast| ──► EWMA smoothing
                                                                          │
                     flagged anomaly sequences ◄── pruning ◄── dynamic threshold ε
```

1. **Forecast.** For each of the 81 labeled channels, a 2-layer LSTM (80 units) reads the last 250 timesteps (telemetry value plus one-hot spacecraft commands) and predicts the next 10 values. It is trained only on the train split, which is mostly nominal.
2. **Error.** On the test split, the absolute prediction error is smoothed with an exponentially weighted moving average (EWMA).
3. **Dynamic threshold.** For each window of errors, the method tries ε = μ + zσ for z = 2.5 … 12. It picks the ε whose removal lowers the mean and standard deviation of the errors the most, while penalising the number of flagged points and sequences. No labels are used.
4. **Pruning.** Flagged sequences whose peak error is not at least 13% higher than the next-highest peak are dropped, which removes small false alarms.
5. **Evaluation (event-level).** A labeled anomaly counts as detected if any flagged sequence overlaps it. A flagged sequence that overlaps no label is a false positive. The scripts report precision, recall, F1 and F0.5, plus recall for **point** and **contextual** anomalies separately.

### Methods

| `--method` | What it is |
|---|---|
| `lstm` | LSTM forecaster with dynamic threshold and pruning (the main model) |
| `lstm_static` | Same LSTM errors with a fixed threshold of mean + 3σ (shows the value of the dynamic threshold) |
| `zscore` | Rolling z-score of the raw signal; threshold = 99.9th percentile on train |
| `iforest` | Isolation Forest on 30-step windows; threshold = 99.9th percentile on train |

## Project layout

```
spacecraft-anomaly/
├── scripts/download_data.py   # fetches the NASA dataset into data/
├── src/sad/
│   ├── config.py              # all hyperparameters (paper defaults)
│   ├── data.py                # loading labels/channels, sliding windows
│   ├── model.py               # LSTMForecaster (PyTorch)
│   ├── train.py               # training with early stopping, forecasting, save/load
│   ├── thresholding.py        # EWMA, dynamic threshold, pruning
│   ├── baselines.py           # z-score, Isolation Forest
│   ├── evaluate.py            # event-level precision / recall / F1
│   ├── run.py                 # CLI: run a method over a spacecraft
│   ├── summarize.py           # mean ± std over seeds
│   └── plot.py                # per-channel plots
├── tests/                     # pytest on synthetic telemetry (no download needed)
├── data/      (created, gitignored)
├── models/    (created, gitignored)
└── results/   (created, gitignored)
```

## 1. Setup

Requires Python 3.10+. A GPU helps but isn't required: each channel's model is small.

```bash
cd spacecraft-anomaly
```

```bash
python -m venv .venv && source .venv/bin/activate
```

On Windows, activate with `.venv\Scripts\activate` instead.

```bash
pip install -e ".[dev]"
```

For MLflow experiment tracking, install `pip install -e ".[dev,mlflow]"` instead. Check that everything works (this uses synthetic data, so no download is needed):

```bash
pytest
```

## 2. Download the data

```bash
python scripts/download_data.py
```

The script tries these sources in turn:

1. **The original NASA/JPL S3 bucket.** It currently returns *403 Access Denied*, but it is tried first in case it comes back.
2. **The public Kaggle mirror** [`patrickfleith/nasa-anomaly-detection-dataset-smap-msl`](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl). This is a direct download of about 86 MB and needs no account.
3. **The same mirror through the Kaggle CLI**, if the direct download fails. Install it with `pip install kaggle` and put your API token in `~/.kaggle/kaggle.json`.
4. **A manual download.** Download the zip from the Kaggle page in your browser, then run:
   ```bash
   python scripts/download_data.py --zip ~/Downloads/archive.zip
   ```

The labels (`labeled_anomalies.csv`) are always downloaded from the [Telemanom GitHub repo](https://github.com/khundman/telemanom). When it finishes you should have:

```
data/labeled_anomalies.csv     # 81 labeled channels: 54 SMAP + 27 MSL, 105 anomalies
data/train/*.npy               # 82 files: per-channel training telemetry
data/test/*.npy                # 82 files: per-channel test telemetry with anomalies
```

To keep the data elsewhere, pass `--data-dir /path` to the commands below or set `SAD_DATA_DIR=/path`.

## 3. Train and evaluate

Start with a quick check on two channels with few epochs:

```bash
python -m sad.run --spacecraft SMAP --channels P-1 E-1 --epochs 5 --run-name smoke
```

Then the full run. This trains one model per channel, saves it to `models/<SPACECRAFT>/seed<N>/`, and writes results to `results/<method>_<SPACECRAFT>_seed<N>/`:

```bash
python -m sad.run --spacecraft SMAP --method lstm --seed 0
```

```bash
python -m sad.run --spacecraft MSL --method lstm --seed 0
```

Saved models are reused automatically, so re-running only re-does thresholding and evaluation. Add `--retrain` to train from scratch.

Run 3 seeds for mean ± std:

```bash
for s in 0 1 2; do for sc in SMAP MSL; do python -m sad.run --spacecraft $sc --method lstm --seed $s; done; done
```

Run the threshold ablation and the baselines. `lstm_static` reuses the trained LSTMs, so there is no extra training.

```bash
for s in 0 1 2; do for sc in SMAP MSL; do for m in lstm_static zscore iforest; do python -m sad.run --spacecraft $sc --method $m --seed $s; done; done; done
```

Build the results table (mean ± std per method and spacecraft):

```bash
python -m sad.summarize results/*_seed*/summary.json
```

Make plots for one channel, or for every channel if you leave the channel out:

```bash
python -m sad.plot results/lstm_SMAP_seed0 P-1
```

Plots are saved to `results/lstm_SMAP_seed0/plots/`.

### Useful options

Every hyperparameter in `src/sad/config.py` can be overridden from the command line, for example `--l-s 250 --n-predictions 10 --hidden-size 80 --epochs 35 --batch-size 64 --lr 0.001 --error-window 2100 --error-buffer 100 --p 0.13`. Other flags:

- `--mlflow` logs parameters, metrics and artifacts to MLflow. View them with `mlflow ui`.
- `--verbose` prints the loss for every epoch.

### Outputs of a run

| File | Contents |
|---|---|
| `summary.json` | TP / FP / FN, precision, recall, F1, F0.5, recall by anomaly class, config |
| `per_channel.csv` | per-channel counts plus true and predicted anomaly sequences |
| `channels/<chan>.npz` | signal, forecast, anomaly score and threshold, used for plots |

## 4. Running on Google Colab

Set **Runtime → Change runtime type → GPU**, then run in a cell:

```python
!git clone https://github.com/424103-achu/fineguard.git
%cd fineguard/spacecraft-anomaly
!pip install -q -e ".[dev]"
!python scripts/download_data.py
!python -m sad.run --spacecraft SMAP --method lstm --seed 0
!python -m sad.run --spacecraft MSL  --method lstm --seed 0
```

To keep models after Colab disconnects, mount Google Drive and pass `--model-dir /content/drive/MyDrive/sad/models --out /content/drive/MyDrive/sad/results`.

## Results

Fill this table only with numbers from your own runs (`python -m sad.summarize`). Don't copy numbers from the paper.

| Method | SMAP P | SMAP R | SMAP F1 | MSL P | MSL R | MSL F1 |
|---|---|---|---|---|---|---|
| lstm (dynamic threshold) | | | | | | |
| lstm_static | | | | | | |
| iforest | | | | | | |
| zscore | | | | | | |

## Notes and caveats

- The labels file lists channel **P-2** twice. `load_labels` merges the two rows, which leaves 54 SMAP channels.
- `T-10` has data files but no labels, so it is skipped.
- Predictions start at test timestep `l_s` (the first 250 steps are history). Flagged sequences are shifted back to test-set indices before they are compared with labels.
- Results differ from the paper because this uses PyTorch rather than Keras and a simplified error-windowing scheme. It also skips the paper's extra pass for unusually *low* errors.
- Wu & Keogh (2021), *"Current Time Series Anomaly Detection Benchmarks are Flawed"*, argue that some anomalies in this dataset are trivial to detect, and that point-adjusted F1 inflates scores. This project reports **event-level** metrics only and includes simple baselines so the comparison is fair.

## References

- K. Hundman, V. Constantinou, C. Laporte, I. Colwell, T. Soderstrom. *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.* KDD 2018.
- Reference implementation and labels: https://github.com/khundman/telemanom
- R. Wu, E. Keogh. *Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress.* IEEE TKDE 2021.
