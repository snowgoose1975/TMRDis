import numpy as np
import pytest

from my_predictor.sequence_features import AMINO_ACIDS, sequence_features


def _reference(sequence: str) -> np.ndarray:
    one = np.zeros((len(sequence), 21), dtype=np.float32)
    indices = [AMINO_ACIDS.index(letter) if letter in AMINO_ACIDS else 20 for letter in sequence]
    one[np.arange(len(sequence)), indices] = 1.0
    sums = np.vstack((np.zeros((1, 21)), one.cumsum(axis=0, dtype=np.float64)))
    starts = np.maximum(np.arange(len(sequence)) - 10, 0)
    ends = np.minimum(np.arange(len(sequence)) + 11, len(sequence))
    local = (sums[ends] - sums[starts]) / (ends - starts)[:, None]
    return np.concatenate((one, local.astype(np.float32)), axis=1)


@pytest.mark.parametrize("sequence", ["A", "ACDEFGHIKLMNPQRSTVWY", "AXBZJUO"])
def test_sequence_features_match_frozen_recipe(sequence):
    actual = sequence_features(sequence)
    np.testing.assert_array_equal(actual, _reference(sequence))
    assert actual.shape == (len(sequence), 42)
    np.testing.assert_allclose(actual[:, :21].sum(axis=1), 1.0)
    np.testing.assert_allclose(actual[:, 21:].sum(axis=1), 1.0, atol=2e-7)


def test_unknown_symbols_share_the_last_category():
    values = sequence_features("AX")
    assert values[1, 20] == 1.0
    assert values[0, 20] == 0.0
    np.testing.assert_allclose(values[0, 41], 0.5)
    np.testing.assert_allclose(values[1, 41], 0.5)


@pytest.mark.parametrize("invalid", ["", "ac", "A C", "A\n"])
def test_sequence_features_reject_invalid_text(invalid):
    with pytest.raises(ValueError):
        sequence_features(invalid)
