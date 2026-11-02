import itertools
import numpy as np
import pytest
from my_predictor.metrics import average_precision


def test_constant_scores_equal_prevalence_for_every_order():
    for labels in set(itertools.permutations([1, 1, 0, 0, 0])):
        assert average_precision(np.ones(5), np.array(labels)) == pytest.approx(0.4)


def test_multiple_tie_groups_are_order_invariant():
    # At score .8 recall increases by 1/2 at precision 1/2;
    # at .2 recall increases by 1/2 at precision 2/3.
    pairs = [(0.8, 1), (0.8, 0), (0.2, 1)]
    for permutation in itertools.permutations(pairs):
        scores, labels = zip(*permutation)
        assert average_precision(scores, labels) == pytest.approx(7 / 12)


def test_untied_scores_match_threshold_definition():
    assert average_precision([.9, .8, .7, .6], [1, 0, 1, 0]) == pytest.approx(5 / 6)


def test_unknown_masked_and_nonfinite_rows_do_not_enter_ties():
    assert average_precision([1, 1, 1, 1, np.nan], [1, 0, 2, 1, 1],
                             [True, True, True, False, True]) == pytest.approx(.5)


def test_no_positive_or_valid_rows_is_undefined():
    assert np.isnan(average_precision([1, 1], [0, 0]))
    assert np.isnan(average_precision([], []))
