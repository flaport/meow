import pytest
from mode_data import MODE_DATA  # type: ignore[reportMissingImports]

from meow import Mode

mode1 = Mode.model_validate(MODE_DATA)
mode2 = Mode.model_validate(MODE_DATA)


def test_multiply_modes():
    """multiplying two modes objects is not supported and should raise a TypeError"""
    try:
        mode1 * mode2  # type: ignore[reportOperatorIssue]
        pytest.fail("Expected TypeError when multiplying two modes objects")
    except TypeError:
        pass


def test_multiply_scalar():
    assert (mode1 * 3).Ex[0, 0] == 3


def test_add_modes():
    assert (mode1 + mode2).Ex[0, 0] == 2


def test_substract_modes():
    assert (mode1 - mode2).Ex[0, 0] == 0


def test_divide_scalar():
    assert (mode1 / 2).Ex[0, 0] == 1 / 2
