from setuptools import setup, Extension
from Cython.Build import cythonize
import os

include_dir = os.path.join(os.path.dirname(__file__), 'include')

# extension_encoder = Extension(
#     name="src.encoder_RM",
#     sources=["src/Encoder/Encoder_RM.cpp", "src/Encoder_RM.pyx"],
#     include_dirs=["include"],
#     extra_compile_args=['std=c11']
# )

extension_decoder = Extension(
    name="PyDecoder_polar",
    sources=["src/Decoder/Decoder_polar.cpp", "src/Decoder/Decoder.cpp", "src/Decoder/PyDecoder_polar.pyx"],
    # include_dirs=["include", "src"],
    include_dirs=[include_dir, os.path.join(include_dir, 'Decoder'), "src/Decoder"],
    language="c++"
)

setup(
    name="PyDecoder_polar",
    ext_modules=cythonize([extension_decoder])
)