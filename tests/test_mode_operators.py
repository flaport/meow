from meow import Mode

MODE_DATA = {
    "neff": {"real": 3.526228477887, "imag": -0.03995016558},
    "Ex": [1, 1],
    "Ey": [2, 2],
    "Ez": [3, 3],
    "Hx": [1, 1],
    "Hy": [2, 2],
    "Hz": [3, 3],
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
}

mode1 = Mode.parse_obj(MODE_DATA)
mode2 = Mode.parse_obj(MODE_DATA)


def test_multiply_modes():
    """multiplying two modes objects is not supported and should raise a TypeError"""
    try:
        mode1 * mode2
        assert False
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


if __name__ == "__main__":
    import pytest  # fmt: skip

    pytest.main([__file__])
