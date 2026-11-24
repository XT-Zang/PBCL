# Constrained MALIS source

The native weight computation in `malis_native.cpp` is a compatibility port of
TuragaLab/malis at commit
`8386d68c87c0a07691c4c0a11b835e0bb33a93db`:

https://github.com/TuragaLab/malis

The original Cython/TensorFlow wrapper does not build with this project's
Python 3.11 and NumPy runtime. The port preserves the maximum-spanning-tree
pair counting algorithm and the positive/negative constrained passes, exposes
one C ABI function, and is compiled into the user's cache on first use.
