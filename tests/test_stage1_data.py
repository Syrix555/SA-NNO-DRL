import numpy as np
import pytest

from op_data_utils import load_op_dataset, normalize_op_instance


def test_load_repository_op_data_uses_common_contract():
    records = load_op_dataset("op_data/unif/op_unif300_len2_seed12345.pkl", num_samples=2)

    assert len(records) == 2
    assert records[0]["loc"].shape == (300, 2)
    assert records[0]["prize"].shape == (300,)
    assert records[0]["depot"].shape == (2,)
    assert records[0]["loc"].dtype == np.float32


@pytest.mark.parametrize(
    "record",
    [
        (np.zeros((4, 2)), np.ones(4), np.array([0.5, 0.5]), 2.0),
        ([0.5, 0.5], np.zeros((4, 2)).tolist(), np.ones(4).tolist(), 2.0),
    ],
)
def test_normalize_historical_tuple_orders(record):
    normalized = normalize_op_instance(record)

    assert normalized["loc"].shape == (4, 2)
    assert normalized["prize"].shape == (4,)
    assert normalized["depot"].shape == (2,)


def test_normalize_rejects_invalid_prize_shape():
    with pytest.raises(ValueError, match="Invalid|prize"):
        normalize_op_instance((np.zeros((4, 2)), np.ones(3), [0.5, 0.5], 2.0))
