import torch
import math

class UncompressTransformLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, compressed_matrix, N, make_symmetric=False, ones_diagonal=False):
        device = compressed_matrix.device
        triu_indices = torch.triu_indices(N, N, offset=1, device=device)
        if make_symmetric:
            sym_indices = torch.stack((torch.cat((triu_indices[0], triu_indices[1])),
                                       torch.cat((triu_indices[1], triu_indices[0]))))
            uncompressed_matrix = (
                torch.sparse_coo_tensor(sym_indices, torch.cat((compressed_matrix, compressed_matrix)),
                                        [N, N])).to_dense()
        else:
            uncompressed_matrix = (torch.sparse_coo_tensor(triu_indices, compressed_matrix, [N, N])).to_dense()

        if ones_diagonal:
            uncompressed_matrix += torch.eye(N, device=device)

        return uncompressed_matrix
