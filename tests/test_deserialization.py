import numpy as np
from mode_data import MODE_DATA

import meow as mw


def test_deserialization():
    simple_mode = mw.Mode.model_validate(MODE_DATA)
    assert np.round(np.real(simple_mode.neff), 6) == np.round(
        MODE_DATA["neff"]["values"][0], 6
    )
    assert np.round(np.imag(simple_mode.neff), 6) == np.round(
        MODE_DATA["neff"]["values"][1], 6
    )
