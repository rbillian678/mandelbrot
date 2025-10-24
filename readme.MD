# Mandelbrot Set Implementation

A high-performance Mandelbrot set implementation in C++ using ARM NEON SIMD instructions, with Python bindings via pybind11 for interactive visualization.

## Features

- **ARM NEON SIMD optimization** for 4x parallel computation
- **Interactive Python visualization** with zoom and pan capabilities
- **Real-time recomputation** when zooming/panning
- **Cross-platform support** (ARM and Intel architectures)

## Requirements

- Python 3.13+
- C++ compiler with NEON support (ARM) or AVX support (Intel)
- pybind11
- matplotlib
- numpy

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Build the C++ extension:**
   ```bash
   python setup.py build_ext --inplace
   ```

## Usage

Run the interactive Mandelbrot visualizer:
```bash
python image.py
```

### Interactive Controls

- **Zoom**: Use the magnifying glass tool in the matplotlib toolbar to drag and select areas to zoom
- **Pan**: Use the pan tool to move around the fractal
- **Reset**: Press 'R' to reset to the default view
- **Save**: Press 'S' to save the current view as `mandelbrot.png`
- **Mouse wheel**: Zoom in/out with mouse wheel

## Architecture-Specific Instructions

### For Intel Processors

If you're running on an Intel processor, you need to modify the code to use AVX instructions instead of NEON:

1. **Replace NEON code with AVX code** in `mandle.cpp`
2. **Update the function call** in `image.py`:
   ```python
   # Change from:
   Z = mandelbrot.mandelbrot_neon(xmin, xmax, ymin, ymax, W, H, it)
   
   # To your new AVX function:
   Z = mandelbrot.your_avx_function(xmin, xmax, ymin, ymax, W, H, it)
   ```

3. **Add the new function binding** at the bottom of `mandle.cpp`:
   ```cpp
   m.def("your_avx_function", &your_avx_function, "Compute iteration count (complex)",
         py::arg("xmin"), py::arg("xmax"), py::arg("ymin"), py::arg("ymax"), 
         py::arg("w"), py::arg("h"), py::arg("max_iter"));
   ```

4. **Update CMakeLists.txt** to detect Intel architecture and enable AVX:
   ```cmake
   # Add after the compiler setting (around line 9)
   # Detect architecture and set appropriate SIMD flags
   if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM|aarch64")
       # ARM architecture - enable NEON
       target_compile_definitions(mandelbrot PRIVATE ARM_NEON)
       target_compile_options(mandelbrot PRIVATE -mfpu=neon)
   elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i386|i686")
       # Intel/AMD architecture - enable AVX
       target_compile_definitions(mandelbrot PRIVATE INTEL_AVX)
       target_compile_options(mandelbrot PRIVATE -mavx -mavx2)
   endif()
   ```

5. **Update setup.py** to detect architecture and set appropriate compile flags:
   ```python
   import platform
   
   # Detect architecture and set appropriate compile flags
   compile_args = ["-std=c++17"]
   if platform.machine().lower() in ['arm64', 'aarch64']:
       compile_args.extend(["-mfpu=neon"])
   elif platform.machine().lower() in ['x86_64', 'amd64']:
       compile_args.extend(["-mavx", "-mavx2"])
   
   ext = Extension(
       name="mandelbrot",
       sources=["mandle.cpp"],
       include_dirs=[pybind11.get_include()],
       language="c++",
       extra_compile_args=compile_args,
   )
   ```

6. **Rebuild the extension:**
   ```bash
   python setup.py build_ext --inplace
   ```

## Project Structure

```
mandlebrot/
├── mandle.cpp          # C++ implementation with NEON SIMD
├── image.py            # Python visualization interface
├── setup.py            # pybind11 build configuration
├── requirements.txt    # Python dependencies
├── CMakeLists.txt      # CMake build configuration
└── README.md           # This file
```

## Performance

The NEON implementation provides significant performance improvements over scalar computation by processing 4 complex numbers simultaneously using SIMD instructions.

## Dependencies

- **pybind11**: C++/Python binding
- **matplotlib**: Visualization
- **numpy**: Numerical computations
- **ARM NEON**: SIMD instructions (ARM processors)
- **AVX**: SIMD instructions (Intel processors)

## Building from Source

The project uses both CMake and pybind11 for building:

```bash
# Using pybind11 (recommended)
python setup.py build_ext --inplace

# Using CMake
mkdir build && cd build
cmake ..
make
```

## License

This project is open source and available under the MIT License.
