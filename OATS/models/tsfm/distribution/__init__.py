# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._base import DistributionOutput, DistrParamProj
from .laplace import LaplaceFixedScaleOutput, LaplaceOutput
from .log_normal import LogNormalOutput
from .mixture import MixtureOutput
from .negative_binomial import NegativeBinomialOutput
from .normal import NormalFixedScaleOutput, NormalOutput
from .pareto import ParetoFixedAlphaOutput, ParetoOutput
from .student_t import StudentTOutput

DISTRIBUTION_OUTPUTS = [
    "LaplaceFixedScaleOutput",
    "LaplaceOutput",
    "LogNormalOutput",
    "MixtureOutput",
    "NegativeBinomialOutput",
    "NormalFixedScaleOutput",
    "NormalOutput",
    "ParetoFixedAlphaOutput",
    "StudentTOutput",
]

__all__ = ["DistrParamProj", "DistributionOutput"] + DISTRIBUTION_OUTPUTS
