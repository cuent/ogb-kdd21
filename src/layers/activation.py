import torch as th


def masked_softmax(
    matrix, mask, dim=-1, memory_efficient=True, mask_fill_value=-1e32
):
    """
    masked_softmax for dgl batch graph
    code snippet contributed by AllenNLP (https://github.com/allenai/allennlp)
    """
    if mask is None:
        result = th.nn.functional.softmax(matrix, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < matrix.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            result = th.nn.functional.softmax(matrix * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_matrix = matrix.masked_fill(
                (1 - mask).byte(), mask_fill_value
            )
            result = th.nn.functional.softmax(masked_matrix, dim=dim)
    return result
