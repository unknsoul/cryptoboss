from src.features.microstructure import microstructure_features


def test_microstructure_features():
    orderbook = {
        "bids": [[100.0, 2.0], [99.5, 1.0]],
        "asks": [[101.0, 1.5], [101.5, 2.0]],
    }

    metrics = microstructure_features(orderbook, atr_value=2.0)

    assert metrics["spread"] == 1.0
    assert round(metrics["book_imbalance"], 3) == round((3.0 - 3.5) / 6.5, 3)
    assert metrics["spread_ratio"] == 0.5
