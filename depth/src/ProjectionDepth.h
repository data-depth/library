/*
  File:             ProjectionDepth.h
  Created by:       Pavlo Mozharovskyi
  First published:  17.05.2013
  Last revised:     13.11.2015
  
  Computation of the projection depth using random sampling.

  For a description of the method, see:
    Zuo, Y.J. and Serfling, R. (2000). General notions of statistical depth
	  function. Annals of Statistics 28, 461-482.
*/

int GetDepthsPrj(TDMatrix points, int n, int d, TDMatrix objects, int m,
	TVariables cardinalities, int k, bool newDirs,
	TDMatrix depths, TDMatrix directions, TDMatrix projections);
