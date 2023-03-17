from os import environ
from sys import stderr
from warnings import filterwarnings

from lightning.fabric.utilities.warnings import PossibleUserWarning
from loguru import logger


def filter_warnings():
    # The dataloader does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument`
    filterwarnings(
        "ignore", category=PossibleUserWarning, module="lightning.pytorch.trainer.connectors.data_connector", lineno=430
    )
    # Total length of `DataLoader` across ranks is zero. Please make sure this was your intention.
    filterwarnings("ignore", category=UserWarning, module="lightning.pytorch.utilities.data", lineno=105)
    # torch.distributed.distributed_c10d._get_global_rank is deprecated please use torch.distributed.distributed_c10d.get_global_rank instead
    filterwarnings("ignore", category=UserWarning, module="torch.distributed.distributed_c10d", lineno=429)
    # Checkpoint directory ... exists and is not empty.
    filterwarnings("ignore", category=UserWarning, module="lightning.pytorch.callbacks.model_checkpoint", lineno=612)
    # torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.
    filterwarnings("ignore", category=UserWarning, module="torch.distributed.distributed_c10d", lineno=2387)
    # torch.distributed._reduce_scatter_base is a private function and will be deprecated. Please use torch.distributed.reduce_scatter_tensor instead.
    filterwarnings("ignore", category=UserWarning, module="torch.distributed.distributed_c10d", lineno=2849)


def setup_logger():
    logger.remove()
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(stderr, format=fmt)


def zero_rank_info(message: str):
    rank = int(environ.get("LOCAL_RANK", 0))
    if rank == 0:
        logger.info(message)
