# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
from pathlib import Path


def load_pkl(path: Path):
    """Load pkl from path."""
    with open(path, "rb") as infile:
        data = pickle.load(infile)
    return data


def save_pkl(data: object, path: Path):
    """Save pkl to path."""
    with open(path, "wb") as outfile:
        pickle.dump(data, outfile)
