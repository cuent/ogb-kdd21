import numpy as np
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors


def smiles2graphft(smiles: str):
    mol = Chem.MolFromSmiles(smiles)

    des_list = [
        "BalabanJ",
        "Kappa1",
        "Chi1",
        "Chi1n",
        "Chi3v",
        "NumHAcceptors",
        "NumRotatableBonds",
        "NumValenceElectrons",
        "Ipc",
        "Kappa2",
        "Kappa3",
        "HallKierAlpha",
        "NumHDonors",
        "HeavyAtomMolWt",
        "HeavyAtomCount",
        "ExactMolWt",
        "Phi",
        "Chi0",
        "LabuteASA",
    ]

    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
    graph_features_list = list(calculator.CalcDescriptors(mol))
    gf = np.array(graph_features_list, dtype=np.float32)
    return gf
