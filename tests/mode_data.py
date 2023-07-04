MODE_DATA = {
    "neff": {"real": 3.526228477887, "imag": -0.03995016558},
    "cs": {
        "structures": [
            {
                "material": {
                    "type": "SampledMaterial",
                    "name": "SiO2",
                    "params": {"wl": [1.2]},
                    "n": {"real": [1.955], "imag": [0.0]},
                },
                "geometry": {
                    "type": "Rectangle",
                    "x_min": -1.0,
                    "x_max": 1.0,
                    "y_min": 0.0,
                    "y_max": 3.5,
                },
                "mesh_order": 5,
            },
            {
                "material": {
                    "type": "SampledMaterial",
                    "name": "SiO2",
                    "params": {"wl": [1.2]},
                    "n": {"real": [1.955], "imag": [0.0]},
                },
                "geometry": {
                    "type": "Rectangle",
                    "x_min": -4.0,
                    "x_max": 4.0,
                    "y_min": -2.0,
                    "y_max": 2.3,
                },
                "mesh_order": 5,
            },
            {
                "material": {
                    "type": "SampledMaterial",
                    "name": "Si",
                    "params": {"wl": [1.2], "T": [38.0]},
                    "n": {"real": [3.526617910662], "imag": [0.0]},
                },
                "geometry": {
                    "type": "Rectangle",
                    "x_min": -0.75,
                    "x_max": 0.75,
                    "y_min": 0.0,
                    "y_max": 3.0,
                },
                "mesh_order": 5,
            },
            {
                "material": {
                    "type": "SampledMaterial",
                    "name": "Si",
                    "params": {"wl": [1.2], "T": [38.0]},
                    "n": {"real": [3.526617910662], "imag": [0.0]},
                },
                "geometry": {
                    "type": "Rectangle",
                    "x_min": -4.0,
                    "x_max": 4.0,
                    "y_min": 0.0,
                    "y_max": 1.8,
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
            "ez_interfaces": False,
        },
        "env": {"wl": 1.2, "T": 38.0},
    },
    "Ex": {"real": [[1.0, 1.0]], "imag": [[0.0, 0.0]]},
    "Ey": {"real": [[2.0, 2.0]], "imag": [[0.0, 0.0]]},
    "Ez": {"real": [[3.0, 3.0]], "imag": [[0.0, 0.0]]},
    "Hx": {"real": [[1.0, 1.0]], "imag": [[0.0, 0.0]]},
    "Hy": {"real": [[2.0, 2.0]], "imag": [[0.0, 0.0]]},
    "Hz": {"real": [[3.0, 3.0]], "imag": [[0.0, 0.0]]},
    "interpolation": None,
}
