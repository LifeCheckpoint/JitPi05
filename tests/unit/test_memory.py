from jitpi05.jitrl.memory import discounted_returns


def test_discounted_returns_preserve_order() -> None:
    assert discounted_returns([1.0, 0.0, 2.0], gamma=0.5) == [1.5, 1.0, 2.0]
