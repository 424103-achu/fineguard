import numpy as np

from sad.data import load_labels, make_input_windows, make_windows, sequences_to_mask


def test_make_windows_shapes_and_targets():
    arr = np.arange(30, dtype=np.float32).reshape(-1, 1).repeat(2, axis=1)
    X, y = make_windows(arr, l_s=5, n_predictions=3)
    assert X.shape == (23, 5, 2) and y.shape == (23, 3)
    assert X[0, -1, 0] == 4 and list(y[0]) == [5, 6, 7]


def test_input_windows_cover_all_future_steps():
    arr = np.zeros((40, 3), dtype=np.float32)
    assert make_input_windows(arr, 10).shape == (30, 10, 3)


def test_mask_is_inclusive():
    m = sequences_to_mask([[2, 4]], 8)
    assert m.tolist() == [False, False, True, True, True, False, False, False]


def test_duplicate_channels_are_merged(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text('chan_id,spacecraft,anomaly_sequences,class,num_values\n'
                 'P-2,SMAP,"[[1, 2]]",[point],10\n'
                 'P-2,SMAP,"[[5, 6]]",[contextual],10\n')
    df = load_labels(p)
    assert len(df) == 1
    assert df.iloc[0].anomaly_sequences == [[1, 2], [5, 6]]
    assert df.iloc[0]["class"] == ["point", "contextual"]
