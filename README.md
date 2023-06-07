# meow

> **M**odeling of **E**igenmodes and **O**verlaps in **W**aveguides

![meow](https://flaport.github.io/meow/_static/meow.png)

A simple electromagnetic [EigenMode Expansion (EME)](https://en.wikipedia.org/wiki/Eigenmode_expansion) tool for Python.

## Installation

### Minimal installation
```sh
pip install meow-sim[min]
```

### Full installation
```sh
pip install meow-sim[full]
```

### Selecting features
You can select which features to enable during installation as follows:
```sh
pip install meow-sim[feature1,feature2,...]
```

#### Available features
* `min`: minimal installation (equivalent to not specifying any features)
* `vis`: explicitly installs matplotlib and trimesh.
* `jax`: use JAX in stead of numpy to calculate the S-matrices
* `klu`: use `klujax` to speed up S-matric calculations (implies `jax`). Note: you need
the SuiteSparse headers on your computer for this to work.
* `gds`: enable GDS creation with GDSFactory (see GDS Taper example).
* `ipy`: install useful jupyter/ipython packages.
* `dev`: install dev dependencies
* `full`: enable all the above features.


## Documentation

- [Examples](https://flaport.github.io/meow/examples.html)
- [API Docs](https://flaport.github.io/meow/meow.html)

## Contributors

- [@flaport](https://gitub.com/flaport): creator of MEOW
- [@jan-david-black](https://github.com/jan-david-black): fixing mode overlaps and more

## Credits

- [DALL-E](https://labs.openai.com): _“a drawing of a kitten with laser eyes walking towards me”_ (logo)
- [Tidy3D](https://github.com/flexcompute/tidy3d): meow uses the free FDE mode solver from Tidy3D.
- [SAX](https://github.com/flaport/sax): meow uses SAX as its circuit simulator when cascading the overlap S-matrices.
- [klujax](https://github.com/flaport/sax): Although technically an optional backend for SAX, klujax will significantly speed up the final S-matrix calculation of your structures.
- [EMEPy](https://github.com/BYUCamachoLab/emepy): an excellent alternative python-based EME solver with optional neural network mode solver.
- [Rigorous and efficient modeling of wavelength scale photonic components](http://photonics.intec.ugent.be/download/phd_104.pdf): PhD thesis of Peter Bienstman.
