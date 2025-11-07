# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .normalized import PackedNMAELoss, PackedNMSELoss, PackedNRMSELoss, PointNormType


class PackedMAELoss(PackedNMAELoss):
    def __init__(self):
        super().__init__(normalize=PointNormType.NONE)


class PackedMSELoss(PackedNMSELoss):
    def __init__(self):
        super().__init__(normalize=PointNormType.NONE)


class PackedRMSELoss(PackedNRMSELoss):
    def __init__(self):
        super().__init__(normalize=PointNormType.NONE)
