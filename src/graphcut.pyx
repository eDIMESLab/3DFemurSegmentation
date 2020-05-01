# distutils: language = c++
# cython: language_level=2
cimport numpy as np
from graphcut cimport GraphCutMaxFlow

def RunGraphCut(unsigned int _totalPixelsInROI,
                np.uint32_t[::1] dataCostPixels,
                np.uint32_t[::1] flat_dataCostSource,
                np.uint32_t[::1] flat_dataCostSink,
                unsigned int _totalNeighbors,
                np.uint32_t[::1] CentersPixels,
                np.uint32_t[::1] NeighborsPixels,
                np.uint32_t[::1] flat_smoothCostFromCenter,
                np.uint32_t[::1] flat_smoothCostToCenter):
  cdef unsigned int * result = GraphCutMaxFlow(_totalPixelsInROI,
                                               & dataCostPixels[0],
                                               & flat_dataCostSource[0],
                                               & flat_dataCostSink[0],
                                               _totalNeighbors,
                                               & CentersPixels[0],
                                               & NeighborsPixels[0],
                                               & flat_smoothCostFromCenter[0],
                                               & flat_smoothCostToCenter[0])
  return [int(result[i]) for i in range(int(_totalPixelsInROI))]
