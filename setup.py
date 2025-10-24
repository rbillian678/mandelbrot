from setuptools import setup, Extension
import pybind11

ext = Extension(
    name="mandelbrot",
    sources=["mandle.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=["-std=c++17"],
)

setup(
    name="mandelbrot",
    version="0.0.1",
    ext_modules=[ext],
)