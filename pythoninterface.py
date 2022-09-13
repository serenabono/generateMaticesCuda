import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray

cudafile = open("generatematrices.cu","r")
mod = SourceModule(cudafile.read(),  options=["--std=c++11", "-DNDEBUG", "-lcurand", "-lcublas" ], no_extern_c=True)
myRand = mod.get_function("generate")
