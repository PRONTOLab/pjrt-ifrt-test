# PjRt/IFRT demo

`src/demo-pjrt.cpp` is a C/C++-translation of what we are doing in Reactant.
The `extern "C"` function are the original functions we have in our support C library, since Julia can only communicate with XLA through the C ABI.
The `main` function is a stripped down version of the orchestration we do in Julia.
