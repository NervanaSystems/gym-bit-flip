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
