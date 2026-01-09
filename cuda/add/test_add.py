import ctypes
from ctypes import c_float, POINTER

lib = ctypes.CDLL("./cuda/add/add.dll") 
rf = ctypes.byref
"""
Build command (rtx 4090)
nvcc --shared add.cu -o add.dll -Xcompiler "/MD" -arch=sm_86
"""

lib.add_cuda.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float)]
lib.add_cuda.restype = None

a = c_float(3.5)
b = c_float(2.25)
result = c_float()

lib.add_cuda(rf(a), rf(b), ctypes.byref(result))

print("Result from CUDA:", result.value)
