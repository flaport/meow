# meow

> **M**odeling of **E**igenmodes and **O**verlaps in **W**aveguides

![meow](https://flaport.github.io/meow/_static/meow.png)

A simple electromagnetic [EigenMode Expansion (EME)](https://en.wikipedia.org/wiki/Eigenmode_expansion) tool for Python.

## Installation

```sh
pip install meow-sim
```

## Documentation

- [Examples](https://flaport.github.io/meow/examples.html)
- [API Docs](https://flaport.github.io/meow/meow.html)

## Credits

- [DALL-E](https://labs.openai.com): _“a drawing of a kitten with laser eyes walking towards me”_ (logo)
- [Tidy3D](https://github.com/flexcompute/tidy3d): meow uses the free FDE mode solver from Tidy3D.
- [SAX](https://github.com/flaport/sax): meow uses SAX as its circuit simulator when cascading the overlap S-matrices.
- [klujax](https://github.com/flaport/sax): Although technically an optional backend for SAX, klujax will significantly speed up the final S-matrix calculation of your structures.
- [EMEPy](https://github.com/emepy): an excellent alternative python-based EME solver with optional neural network mode solver.
- [Rigorous and efficient modeling of wavelength scale photonic components](http://photonics.intec.ugent.be/download/phd_104.pdf): PhD thesis of Peter Bienstman.
