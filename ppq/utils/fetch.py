from numpy import dtype
import torch

def random_fetch(tensor: torch.Tensor, num_of_elements: int = 1024):
    """
    Fetch some elements from tensor randomly.

    Args:
        tensor (torch.Tensor): [description]
        num_of_elements (int, optional): [description]. Defaults to 1024.
    """
    
    def generate_indexer(num_of_elements: int, mod: int, seed: int = 0x20211230) -> torch.Tensor:
        indexer = []
        for i in range(num_of_elements):
            indexer.append(seed % mod)
            seed = (0x343FD * seed + 0x269EC3) % (2 << 31)
        return torch.tensor(indexer, dtype=torch.int32)
    
    total_elements = tensor.numel()
    assert total_elements > 0, ('Can not fetch data from tensor with less than 1 elements.')

    indexer = generate_indexer(num_of_elements=num_of_elements, mod=total_elements)
    return tensor.flatten().index_select(dim=0, index=indexer.to(tensor.device))