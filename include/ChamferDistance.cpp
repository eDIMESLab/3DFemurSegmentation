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
  long zy;
  z = shapeZ + 1L;
  y = shapeY + 1L;
  x = shapeX + 1L;
  zy = z*y;
  long d, dmw, dmwh, wmh, h, hm, w, wm;
  for ( long i=1L; i<z; ++i ) {
    d = i*zy;
    for ( long j=1L; j<y; ++j ) {
      w = d + j*y;
      wm = w - y;
      dmw = w - zy;
      for ( long k=1L; k<x; ++k ) {
        h = w + k; // position: k + j*y + i*zy
        hm = h - 1L; // position: (k-1) + j*y + i*zy
        wmh = wm + k; // position: k + (j-1)*y + i*zy
        dmwh = dmw + k; // position: k + j*y + (i-1)*zy
        im[h] = std :: min({im[h],
                            im[hm] + weight,
                            im[wmh] + weight,
                            im[dmwh] + weight});
      }
    }
  }

  for ( long i=z-1L; i>0L; --i ) {
    d = i*zy;
    for ( long j=y-1L; j>0L; --j ) {
      w = d + j*y;
      wm = w + y;
      dmw = w + zy;
      for ( long k=x-1L; k>0L; --k ) {
        h = w + k; // position: k + j*y + i*zy
        hm = h + 1L; // position: (k+1) + j*y + i*zy
        wmh = wm + k; // position: k + (j+1)*y + i*zy
        dmwh = dmw + k; // position: k + j*y + (i+1)*zy
        im[h] = std :: min({im[h],
                            im[hm] + weight,
                            im[wmh] + weight,
                            im[dmwh] + weight});
      }
    }
  }

  return im;
}
