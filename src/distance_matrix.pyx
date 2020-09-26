# distutils: language = c++
# cython: language_level=2
cimport numpy as np
from chamferdistance cimport ManhattanChamferDistance

def ComputeChamferDistance(long [::1] _im,
                           unsigned int _L,
                           long _shapeZ,
                           long _shapeY,
                           long _shapeX,
                           long _weight):

  cdef long * result = ManhattanChamferDistance(& _im[0],
                                                _L,
                                                _shapeZ,
                                                _shapeY,
                                                _shapeX,
                                                _weight)
  return [int(result[i]) for i in range(_L)]
