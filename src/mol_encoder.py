import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims


def get_new_atom_feature_dims():
    return [
        len([0, 1, 2, 3, 4, 5, 6]),  # explicit_valence
        len([0, 1, 2, 3, 4]),  # implicit_valence
        len(list(range(0, 90))),  # isotopes
        len(list(range(0, 127))),  # mass
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
