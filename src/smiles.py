from abc import ABC, abstractmethod
from typing import Any

from ogb.utils import smiles2graph


class Smiles2GraphConverter(ABC):
    @abstractmethod
    def __call__(self, smiles: str) -> Any:
        pass


class Smiles2GraphOGBConverter(Smiles2GraphConverter):
    def __call__(self, smiles: str) -> Any:
        return smiles2graph(smiles)


class Smiles2GraphFeatures(Smiles2GraphConverter):
    def __call__(self, smiles: str) -> Any:
        return smiles2graph(smiles)
