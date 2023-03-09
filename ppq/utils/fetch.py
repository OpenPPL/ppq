import torch


def generate_indexer(
    num_of_fetches: int,
    num_of_elements: int,
    seed: int = 0x20211230) -> torch.Tensor:
    """Sample with a given seed. This function will generates a indexer based
    on your seed.

    Args:
        num_of_fetches (int): [description]
        num_of_elements (int): [description]
        seed (int, optional): [description]. Defaults to 0x20211230.

    Returns:
        torch.Tensor: [description]
    """

    indexer = []
    for i in range(num_of_fetches):
        indexer.append(seed % num_of_elements)
        seed = (0x343FD * seed + 0x269EC3) % (2 << 31)
    return torch.tensor(indexer, dtype=torch.int32)

def generate_torch_indexer(
    num_of_fetches: int,
    num_of_elements: int) -> torch.Tensor:
    return torch.randint(low=0, high=num_of_elements, size=[num_of_fetches])


def tensor_random_fetch(
    tensor: torch.Tensor, seed: int = None,
    num_of_fetches: int = 1024) -> torch.Tensor:
    """Fetch some elements from tensor randomly. if a valid seed is given,
    elements will be sampled based on your seed, otherwise a random seed will
    be generated.

    Args:
        tensor (torch.Tensor): [description]
        num_of_fetches (int, optional): [description]. Defaults to 1024.
    """
    tensor = tensor.flatten()
    num_of_elements = tensor.numel()
    assert num_of_elements > 0, ('Can not fetch data from empty tensor(0 element).')

    if seed is None:
        indexer = generate_torch_indexer(num_of_fetches=num_of_fetches, num_of_elements=num_of_elements)
    else: indexer = generate_indexer(num_of_fetches=num_of_fetches, num_of_elements=num_of_elements, seed=seed)
    return tensor.index_select(dim=0, index=indexer.to(tensor.device).long())


def channel_random_fetch(
    tensor: torch.Tensor,
    fetchs_per_channel: int = 1024,
    seed: int = None,
    channel_axis: int = 0) -> torch.Tensor:
    """Fetch some elements from tensor randomly by each channel. if a valid
    seed is given, elements will be sampled based on your seed, otherwise a
    random seed will be generated.

    result: [num_of_channel, fetchs_per_channel]

    Args:
        tensor (torch.Tensor): [description]
        fetchs_per_channel (int, optional): [description]. Defaults to 1024.
        channel_axis (int, optional): [description]. Defaults to 0.

    Returns:
        torch.Tensor: [description]
    """
    tensor = tensor.transpose(0, channel_axis)
    tensor = tensor.flatten(start_dim=1)
    num_of_elements = tensor.shape[-1]
    assert num_of_elements > 0, ('Can not fetch data from empty tensor(0 element).')

    if seed is None:
        indexer = generate_torch_indexer(num_of_fetches=fetchs_per_channel, num_of_elements=num_of_elements)
    else: indexer = generate_indexer(num_of_fetches=fetchs_per_channel, num_of_elements=num_of_elements, seed=seed)
    return tensor.index_select(dim=-1, index=indexer.to(tensor.device).long())


def batch_random_fetch(
    tensor: torch.Tensor,
    fetches_per_batch: int = 1024,
    seed: int = None
    ) -> torch.Tensor:
    """Fetch some elements from each sample in a batched tensor. if a valid
    seed is given, elements will be sampled based on your seed, otherwise a
    random seed will be generated.

    result: [num_of_batch, fetchs_per_batch]

    Args:
        tensor (torch.Tensor): [description]
        fetchs_per_channel (int, optional): [description]. Defaults to 1024.

    Returns:
        torch.Tensor: [description]
    """
    tensor = tensor.flatten(start_dim=1)
    num_of_elements = tensor.shape[-1]
    assert num_of_elements > 0, ('Can not fetch data from empty tensor(0 element).')

    if seed is None:
        indexer = generate_torch_indexer(num_of_fetches=fetches_per_batch, num_of_elements=num_of_elements)
    else: indexer = generate_indexer(num_of_fetches=fetches_per_batch, num_of_elements=num_of_elements, seed=seed)
    return tensor.index_select(dim=-1, index=indexer.to(tensor.device).long())
