# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import warnings
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def get_path_var(var: Optional[str]) -> Optional[Path]:
    if (path := os.getenv(var)) is not None:
        return Path(path)
    return None


class Env:
    _instance: Optional["Env"] = None
    path_vars: list[str] = [
        "LOTSA_10M_PATH",
        "LOTSA_100M_PATH",
        "LOTSA_1B_PATH",
        "LOTSA_16B_PATH",
        "LOTSA_V1_PATH",
        "CUSTOM_DATA_PATH"
    ]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if not load_dotenv():
                warnings.warn("Failed to load .env file.")
            cls.monkey_patch_path_vars()
        return cls._instance

    @classmethod
    def monkey_patch_path_vars(cls):
        for var in cls.path_vars:
            setattr(cls, var, get_path_var(var))


env = Env()