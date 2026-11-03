"""Hyperparameters. Defaults follow the Telemanom paper / reference config."""

from dataclasses import asdict, dataclass


@dataclass
class Config:
    # windowing
    l_s: int = 250              # history length fed to the LSTM
    n_predictions: int = 10     # forecast horizon (steps ahead)

    # model
    hidden_size: int = 80
    num_layers: int = 2
    dropout: float = 0.3

    # training
    epochs: int = 35
    batch_size: int = 64
    lr: float = 1e-3
    validation_split: float = 0.2   # last 20% of train windows (chronological)
    patience: int = 10
    min_delta: float = 3e-4

    # error processing / thresholding
    smoothing_perc: float = 0.05    # EWMA span = smoothing_perc * error_window
    error_window: int = 2100        # errors are thresholded in windows of this length
    error_buffer: int = 100         # pad flagged points before grouping into sequences
    p: float = 0.13                 # pruning: min % drop between consecutive max errors
    z_range: tuple = (2.5, 12.0, 0.5)

    def to_dict(self) -> dict:
        return asdict(self)
