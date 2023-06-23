import numpy as np
from pydantic import Field

import meow as mw


class SimpleMode(mw.BaseModel):
    neff: complex = Field(description="the effective index of the mode")
    cs: mw.CrossSection = Field(
        description="the index cross section for which the mode was calculated"
    )
    te_fraction: float = Field(description="the TE polarization fraction of the mode.")

    @property
    def env(self):
        return self.cs.env

    @property
    def mesh(self):
        return self.cs.mesh

    @property
    def cell(self):
        return self.cs.cell


SIMPLE_MODE_DATA = {
    "neff": {"real": 3.526228477887, "imag": -0.03995016558},
    "cs": {
        "cell": {
            "structures": [
                {
                    "material": {
                        "name": "SiO2",
                        "params": {"wl": [1.2]},
                        "n": {"real": [1.955], "imag": [0.0]},
                        "type": "SampledMaterial",
                    },
                    "geometry": {
                        "type": "Box",
                        "x_min": -1.0,
                        "x_max": 1.0,
                        "y_min": 0.0,
                        "y_max": 3.5,
                        "z_min": -1.0,
                        "z_max": 1.0,
                    },
                    "mesh_order": 5,
                },
                {
                    "material": {
                        "name": "SiO2",
                        "params": {"wl": [1.2]},
                        "n": {"real": [1.955], "imag": [0.0]},
                        "type": "SampledMaterial",
                    },
                    "geometry": {
                        "type": "Box",
                        "x_min": -4.0,
                        "x_max": 4.0,
                        "y_min": -2.0,
                        "y_max": 2.3,
                        "z_min": -1.0,
                        "z_max": 1.0,
                    },
                    "mesh_order": 5,
                },
                {
                    "material": {
                        "name": "Si",
                        "params": {"wl": [1.2], "T": [38.0]},
                        "n": {"real": [3.526617910662], "imag": [0.0]},
                        "type": "SampledMaterial",
                    },
                    "geometry": {
                        "type": "Box",
                        "x_min": -0.75,
                        "x_max": 0.75,
                        "y_min": 0.0,
                        "y_max": 3.0,
                        "z_min": -1.0,
                        "z_max": 1.0,
                    },
                    "mesh_order": 5,
                },
                {
                    "material": {
                        "name": "Si",
                        "params": {"wl": [1.2], "T": [38.0]},
                        "n": {"real": [3.526617910662], "imag": [0.0]},
                        "type": "SampledMaterial",
                    },
                    "geometry": {
                        "type": "Box",
                        "x_min": -4.0,
                        "x_max": 4.0,
                        "y_min": 0.0,
                        "y_max": 1.8,
                        "z_min": -1.0,
                        "z_max": 1.0,
                    },
                    "mesh_order": 5,
                },
            ],
            "mesh": {
                "x": [],
                "y": [],
                "angle_phi": 0.0,
                "angle_theta": 0.0,
                "bend_radius": 6000.0,
                "bend_axis": 1,
                "num_pml": [6, 6],
            },
            "z_min": -1.0,
            "z_max": -1.0,
        },
        "env": {"wl": 1.2, "T": 38.0},
    },
    "te_fraction": 0.9998199906370098,
}


def test_deserialization():
    simple_mode = SimpleMode.parse_obj(SIMPLE_MODE_DATA)
    assert np.round(np.real(simple_mode.neff), 6) == np.round(
        SIMPLE_MODE_DATA["neff"]["real"], 6
    )
    assert np.round(np.imag(simple_mode.neff), 6) == np.round(
        SIMPLE_MODE_DATA["neff"]["imag"], 6
    )
