import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims


def get_new_atom_feature_dims():
    return [
        len([0, 1, 2, 3, 4, 5, 6]),  # explicit_valence
        len([[0, 1, 2, 3, 4]]),  # implicit_valence
        len(
            [
                0,
                2,
                35,
                14,
                18,
                11,
                19,
                13,
                3,
                15,
                17,
                10,
                38,
                12,
                16,
                1,
                76,
                32,
                8,
                20,
                21,
                22,
                33,
                9,
                24,
                25,
                26,
                36,
                79,
                82,
                34,
                81,
                74,
                37,
                80,
                75,
                31,
                77,
                72,
                40,
                84,
                89,
                63,
                62,
                83,
                39,
                41,
                85,
                65,
            ]
        ),  # isotopes
        len(
            (
                [
                    12,
                    14,
                    15,
                    18,
                    35,
                    32,
                    79,
                    10,
                    72,
                    28,
                    30,
                    2,
                    78,
                    74,
                    34,
                    22,
                    11,
                    19,
                    13,
                    3,
                    16,
                    17,
                    37,
                    1,
                    126,
                    75,
                    31,
                    8,
                    20,
                    21,
                    40,
                    24,
                    47,
                    9,
                    25,
                    26,
                    69,
                    65,
                    81,
                    33,
                    80,
                    73,
                    39,
                    4,
                    36,
                    76,
                    50,
                    55,
                    71,
                    83,
                    88,
                    62,
                    61,
                    82,
                    38,
                    6,
                    44,
                    84,
                    64,
                    63,
                    58,
                ]
            )
        ),  # mass
        len([0, 1]),  # no_implicit,
        len([0, 1, 2, 3, 4]),  # explicit_hs
        len([0, 1, 2, 3, 4, 5, 6]),  # total_valence
    ]


def get_enchanced_dims():
    return [*get_atom_feature_dims(), *get_new_atom_feature_dims()]


full_atom_feature_dims = get_enchanced_dims()
full_bond_feature_dims = get_bond_feature_dims()


class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding
