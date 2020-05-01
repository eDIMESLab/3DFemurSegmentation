# distutils: language = c++
# cython: language_level=2
cimport numpy as np


def ManhattanChamferDistance(np.ndarray im, tuple shape, float w=1.):
  cdef long z
  cdef long y
  cdef long x
  z, y, x = shape[0], shape[1], shape[2]
  z += 1
  y += 1
  x += 1
  cdef long i
  cdef long j
  cdef long k
  cdef float pixel
  for i in range(1, z):
    for j in range(1, y):
      for k in range(1, x):
        pixel = min(im[i - 1, j, k] + w,
                    im[i, j - 1, k] + w,
                    im[i, j, k - 1] + w,
                    im[i, j, k]
                    )
        im[i, j, k] = pixel

  for i in reversed(range(1, z)):
    for j in reversed(range(1, y)):
      for k in reversed(range(1, x)):
        pixel = min(im[i + 1, j, k] + w,
                    im[i, j + 1, k] + w,
                    im[i, j, k + 1] + w,
                    im[i, j, k]
                    )
        im[i, j, k] = pixel
  return im
