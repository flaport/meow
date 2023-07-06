import numpy as np
from mode_data import MODE_DATA

import meow as mw


def test_deserialization():
    simple_mode = mw.Mode.parse_obj(MODE_DATA)
    assert np.round(np.real(simple_mode.neff), 6) == np.round(
        MODE_DATA["neff"]["real"], 6
    )
    assert np.round(np.imag(simple_mode.neff), 6) == np.round(
        MODE_DATA["neff"]["imag"], 6
    )
