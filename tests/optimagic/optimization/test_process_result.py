from optimagic.optimization.process_results import switch_sign


def test_switch_sign_dict():
    d = {"contributions": 1, "value": -1}
    calculated = switch_sign(d)
    expected = {"contributions": -1, "value": 1}
    assert calculated == expected
