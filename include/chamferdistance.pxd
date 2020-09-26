# distutils: language = c++
# cython: language_level=2

cdef extern from "ChamferDistance.cpp":
  cdef long * ManhattanChamferDistance ( long * im,
                                         long shapeZ,
                                         long shapeY,
                                         long shapeX,
                                         long weight
                                       )
