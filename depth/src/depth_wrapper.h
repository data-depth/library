#include "ProjectionDepths.h"
#include <iterator>

extern "C"
{

int SetDepthPars(cProjection& depthObj, int n_refinements,
				double sphcap_shrink, double alpha_Dirichlet,
				double cooling_factor, double cap_size,
				int start, int line_solver, int bound_gc);


int depth_approximation(double *z, double *x, int notion, int solver,
						int NRandom, int option, int n_refinements,
						double sphcap_shrink, double alpha_Dirichlet,
						double cooling_factor,double cap_size, int start,
						int line_solver, int bound_gc, int n, int d,
						int n_z, double *depths, double *depths_iter,
						double *directions, int *directions_card);

};