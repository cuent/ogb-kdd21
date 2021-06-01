from abc import ABC, abstractmethod
from typing import Dict, Any

from ogb.utils import smiles2graph


class Smiles2GraphConverter(ABC):
    @abstractmethod
    def __call__(self, smiles: str) -> Dict[str, Any]:
        pass


class Smiles2GraphOGBConverter(Smiles2GraphConverter):
    def __call__(self, smiles: str) -> Dict[str, Any]:
        return smiles2graph(smiles)
