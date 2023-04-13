from itertools import islice
from os import environ

from torch.utils.data import IterableDataset, get_worker_info

from src.utils import zero_rank_info


class DistributedIterableDataset(IterableDataset):
    """A wrapper over a classic iterable dataset for correct work with multiples workers across multiple devices.

    Separates data between ranks in the case of training on several gpus and
    between workers in the case of using a dataloader with more than 1 worker.
    """

    def __init__(self, dataset: IterableDataset):
        self.dataset = dataset

    def __iter__(self):
        world_size, rank = int(environ.get("WORLD_SIZE", 1)), int(environ.get("LOCAL_RANK", 0))
        num_workers, worker_id = 1, 0

        worker_info = get_worker_info()
        if worker_info is not None:
            num_workers, worker_id = worker_info.num_workers, worker_info.id

        if worker_id == 0:
            zero_rank_info(f"Distributed Dataset setting: {world_size} devices, {num_workers} workers.")

        total_chunks = world_size * num_workers
        chunk_id = (rank * num_workers) + worker_id

        yield from islice(self.dataset, chunk_id, None, total_chunks)
