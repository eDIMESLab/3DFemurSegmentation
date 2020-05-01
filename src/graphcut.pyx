# distutils: language = c++
# cython: language_level=2
cimport numpy as np
from graphcut cimport GraphCutMaxFlow

def RunGraphCut(unsigned int _totalPixelsInROI,
                np.uint32_t[::1] dataCostPixels,
                float[::1] flat_dataCostSource,
                float[::1] flat_dataCostSink,
                unsigned int _totalNeighbors,
                np.uint32_t[::1] NeighborsPixels,
                float[::1] flat_smoothCostFromCenter,
                float[::1] flat_smoothCostToCenter):
  cdef unsigned int * result = GraphCutMaxFlow(_totalPixelsInROI,
                                               & dataCostPixels[0],
                                               & flat_dataCostSource[0],
                                               & flat_dataCostSink[0],
                                               _totalNeighbors,
                                               & NeighborsPixels[0],
                                               & flat_smoothCostFromCenter[0],
                                               & flat_smoothCostToCenter[0])
  return [int(result[i]) for i in range(int(_totalPixelsInROI))]
