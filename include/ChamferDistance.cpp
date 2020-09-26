#include <iostream>
#include <algorithm>

long * ManhattanChamferDistance ( long * im,
                                  long shapeZ,
                                  long shapeY,
                                  long shapeX,
                                  long weight=1L
                                )
{
  long z;
  long y;
  long x;
  long xy;
  z = shapeZ + 1L;
  y = shapeY + 1L;
  x = shapeX + 1L;
  xy = x*y;
  long d, dmw, dmwh, wmh, h, hm, w, wm;
  long pixel;
  for ( long i=1L; i<z; ++i ) {
    d = i*xy;
    for ( long j=1L; j<y; ++j ) {
      w = d + j*x;
      wm = w - x;
      dmw = w - xy;
      for ( long k=1L; k<x; ++k ) {
        h = w + k; // position: k + j*x + i*xy
        hm = h - 1L; // position: (k-1) + j*x + i*xy
        wmh = wm + k; // position: k + (j-1)*x + i*xy
        dmwh = dmw + k; // position: k + j*x + (i-1)*xy
        pixel = std :: min({im[h],
                            im[hm] + weight,
                            im[wmh] + weight,
                            im[dmwh] + weight});
        im[h] = pixel;
      }
    }
  }

  for ( long i=z-1L; i>0L; --i ) {
    d = i*xy;
    for ( long j=y-1L; j>0L; --j ) {
      w = d + j*x;
      wm = w + x;
      dmw = w + xy;
      for ( long k=x-1L; k>0L; --k ) {
        h = w + k; // position: k + j*x + i*xy
        hm = h + 1L; // position: (k+1) + j*x + i*xy
        wmh = wm + k; // position: k + (j+1)*x + i*xy
        dmwh = dmw + k; // position: k + j*x + (i+1)*xy
        pixel = std :: min({im[h],
                            im[hm] + weight,
                            im[wmh] + weight,
                            im[dmwh] + weight});
        im[h] = pixel;
      }
    }
  }

  return im;
}
