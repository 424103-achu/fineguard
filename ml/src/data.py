"""Loading UNFAIR-ToS (LexGLUE) and converting labels to multi-hot matrices."""
import numpy as np

DATASET = ("coastalcph/lex_glue", "unfair_tos")
SPLITS = ("train", "validation", "test")


def load_unfair_tos():
    """Load UNFAIR-ToS and return (dataset_dict, label_names).

    The label order is read from the dataset features, never hardcoded.
    The chronological train/validation/test split is kept as released.
    """
    from datasets import load_dataset

    ds = load_dataset(*DATASET)
    label_names = list(ds["train"].features["labels"].feature.names)
    return ds, label_names


def to_multi_hot(label_lists, n_labels):
    """Convert [[0, 3], [], [5]] into a (n, n_labels) 0/1 matrix. Empty list = fair."""
    y = np.zeros((len(label_lists), n_labels), dtype=np.int64)
    for i, labs in enumerate(label_lists):
        for j in labs:
            y[i, j] = 1
    return y


def get_split(ds, split, n_labels):
    texts = list(ds[split]["text"])
    y = to_multi_hot(ds[split]["labels"], n_labels)
    return texts, y


def load_all_splits():
    """Return (splits, label_names) with splits[name] = (texts, y)."""
    ds, labels = load_unfair_tos()
    return {s: get_split(ds, s, len(labels)) for s in SPLITS}, labels
