import numpy as np
import pytest

from gym_bit_flip import BitFlip


def test_bitflip_1():
    bit_flip = BitFlip(1)
    bit_flip.state = np.array([0])
    bit_flip._step(0)

    np.testing.assert_array_equal(bit_flip.state, np.array([1]))


def test_bitflip_2():
    bit_flip = BitFlip(2)
    bit_flip.state = np.array([0, 0])
    bit_flip._step(1)

    np.testing.assert_array_equal(bit_flip.state, np.array([0, 1]))


def test_bitflip_not_terminate_long():
    bit_flip = BitFlip(256)
    assert bit_flip._terminate() is False


def test_bitflip_not_terminate_short():
    bit_flip = BitFlip(2)
    for _ in range(16):
        assert bit_flip._terminate() is False
        bit_flip.reset()


def test_bitflip_bit_length_0():
    with pytest.raises(ValueError):
        BitFlip(bit_length=0)


def test_reward():
    bit_flip = BitFlip(2)
    bit_flip.state = np.array([0, 0])
    bit_flip.goal = np.array([1, 0])
    _, reward, _, _ = bit_flip._step(1)

    assert reward == -1


def test_reward():
    bit_flip = BitFlip(256)
    _, reward, _, _ = bit_flip._step(0)

    assert reward == -1


def test_mean_zero():
    bit_flip = BitFlip(mean_zero=True)
    state, _, _, _ = bit_flip._step(0)

    assert 0.5 in state['state']
    assert -0.5 in state['state']
    assert 0.5 in state['goal']
    assert -0.5 in state['goal']
