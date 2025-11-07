# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from dataclasses import dataclass
from typing import Any


class Transformation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]: ...

    def chain(self, other: "Transformation") -> "Chain":
        return Chain([self, other])

    def __add__(self, other: "Transformation") -> "Chain":
        return self.chain(other)

    def __radd__(self, other):
        if other == 0:
            return self
        return other + self


@dataclass
class Chain(Transformation):
    """
    Chain multiple transformations together.
    """

    transformations: list[Transformation]

    def __post_init__(self) -> None:
        transformations = []

        for transformation in self.transformations:
            if isinstance(transformation, Identity):
                continue
            elif isinstance(transformation, Chain):
                transformations.extend(transformation.transformations)
            else:
                assert isinstance(transformation, Transformation)
                transformations.append(transformation)

        self.transformations = transformations
        self.__init_passed_kwargs__ = {"transformations": transformations}

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        for t in self.transformations:
            data_entry = t(data_entry)
        return data_entry


class Identity(Transformation):
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        return data_entry
