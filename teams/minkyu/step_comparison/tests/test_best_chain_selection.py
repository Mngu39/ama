from puyo_train import is_better_chain


def test_best_chain_selection_priority():
    assert is_better_chain((6.5, 30000), None)
    assert is_better_chain((6.6, 25000), (6.5, 40000))
    assert is_better_chain((6.5, 31000), (6.5, 30000))
    assert not is_better_chain((6.4, 80000), (6.5, 30000))
    assert not is_better_chain((6.5, 30000), (6.5, 30000))
