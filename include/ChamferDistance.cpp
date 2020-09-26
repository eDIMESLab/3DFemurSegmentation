#include <iostream>
#include <algorithm>

long * ManhattanChamferDistance ( long * im,
                                  long shapeZ,
                                  long shapeY,
                                  long shapeX,
                                  long weight=1L
                                )
{
  long depth;
  long height;
  long width, widthplus;
  long area;
  depth  = shapeZ + 1L;
  height = shapeY + 1L;
  width  = shapeX + 1L;
  widthplus = width + 1L;
  area   = widthplus*(height+1);
  long trail_j, trail_k, trail_jk;
  long pos, pos_i, pos_j, pos_k;
  long pixel;
  for ( long k=1L; k<depth; ++k ) {
    trail_k = k*area;
    for ( long j=1L; j<height; ++j ) {
      trail_j = j*widthplus;
      trail_jk = trail_j + trail_k;
      for ( long i=1L; i<width; ++i ) {
        pos = i + trail_jk;  // [i,j,k] == i + j*width + k*area
        pos_i = pos - 1;     // [i-1,j,k] == (i-1) + j*width + k*area
        pos_j = pos - widthplus; // [i,j-1,k] == i + (j-1)*width + k*area
        pos_k = pos - area;  // [i,j,k-1] == i + j*width + (k-1)*area
        pixel = std :: min({im[pos],
                            im[pos_i] + weight,
                            im[pos_j] + weight,
                            im[pos_k] + weight});
        im[pos] = pixel;
      }
    }
  }

  for ( long k=shapeZ; k>0; --k ) {
    trail_k = k*area;
    for ( long j=shapeY; j>0; --j ) {
      trail_j = j*width;
      trail_jk = trail_j + trail_k;
      for ( long i=shapeX; i>0; --i ) {
        pos = i + trail_jk;  // [i,j,k] == i + j*width + k*area
        pos_i = pos + 1;     // [i-1,j,k] == (i+1) + j*width + k*area
        pos_j = pos + widthplus; // [i,j-1,k] == i + (j+1)*width + k*area
        pos_k = pos + area;  // [i,j,k-1] == i + j*width + (k+1)*area
        pixel = std :: min({im[pos],
                            im[pos_i] + weight,
                            im[pos_j] + weight,
                            im[pos_k] + weight});
        im[pos] = pixel;
      }
    }
  }

  return im;
}
