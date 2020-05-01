# distutils: language = c++
# cython: language_level=2

cdef extern from "3DGraphCut.cpp":
  cdef unsigned int * GraphCutMaxFlow ( int &_totalPixelsInROI,
                                        unsigned int * dataCostPixels,
                                        float * flat_dataCostSource,
                                        float * flat_dataCostSink,
                                        unsigned int &_totalNeighbors,
                                        unsigned int * NeighborsPixels,
                                        float * flat_smoothCostFromCenter,
                                        float * flat_smoothCostToCenter
                                       )
