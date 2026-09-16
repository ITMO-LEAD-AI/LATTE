from latte.rtdetr_utils import select_layers_by_score


def test_selects_layers_with_lowest_combined_score():
    checkpoint = {
        "layers": {
            0: {"cosine_loss": 0.01, "mse_loss": 0.10},
            1: {"cosine_loss": 0.20, "mse_loss": 0.01},
            2: {"cosine_loss": 0.05, "mse_loss": 0.02},
        }
    }
    assert select_layers_by_score(checkpoint, 2, mse_weight=10.0) == [1, 2]


def test_zero_layers_returns_empty_selection():
    checkpoint = {"layers": {0: {"cosine_loss": 0.1, "mse_loss": 0.1}}}
    assert select_layers_by_score(checkpoint, 0) == []
