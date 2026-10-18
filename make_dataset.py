"""Legacy SA dataset entry point using the shared OP data contract."""

from op_data_utils import load_op_dataset
from utils.data_utils import check_extension, save_dataset


def make_dataset(filename=None, num_samples=100, offset=0):
    if filename is None:
        raise ValueError("An OP dataset filename is required")
    return load_op_dataset(filename, num_samples=num_samples, offset=offset)
