

void OjaDepthsEx(TDMatrix X, TDMatrix x, int d, int n, int nx, int useCov,
                 TDMatrix covEst, double *depths);

void OjaDepthsApx(TDMatrix X, TDMatrix x, int d, int n, int nx,
	unsigned long long k, int useCov, TDMatrix covEst, double *depths);
