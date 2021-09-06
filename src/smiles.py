from abc import ABC, abstractmethod
from typing import Any

from ogb.utils import smiles2graph

from src.converters import smiles2graph_enchanced


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


class Smiles2GraphOGBEnchanced(Smiles2GraphConverter):
    def __call__(self, smiles: str) -> Any:
        return smiles2graph_enchanced(smiles)
