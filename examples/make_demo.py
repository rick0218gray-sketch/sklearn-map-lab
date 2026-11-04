"""Write a deterministic classification CSV with synthetic data."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
features, target = make_classification(
    n_samples=120, n_features=5, n_informative=3, random_state=42
)
frame = pd.DataFrame(features, columns=["feature_a", "feature_b", "feature_c", "feature_d", "feature_e"])
frame["target"] = target
frame.to_csv(args.output, index=False)
