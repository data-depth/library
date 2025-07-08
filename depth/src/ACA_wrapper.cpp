#include "APD.cpp"
#include "auxLinAlg.cpp"
#include "HD.cpp"
#include "Matrix.cpp"
#include "MD.cpp"
#include "mvrandom.cpp"
#include "PD.cpp"
#include "ProjectionDepths.cpp"
#include "ZD.cpp"
#include "WMD.cpp"
#include <iterator>

extern "C"
{

int SetDepthPars(cProjection& depthObj, int n_refinements,
				double sphcap_shrink, double alpha_Dirichlet,
				double cooling_factor, double cap_size,
				int start, int line_solver, int bound_gc, int option){
	
	// Set cProjection variables
	depthObj.SetOption(option);
	switch(depthObj.GetMethod()){
	case eProjMeth::RefinedRandom:
		depthObj.SetMaxRefinesRand(n_refinements);
		depthObj.SetAlphaRand(sphcap_shrink);
		break;
	case eProjMeth::RefinedGrid:
		depthObj.SetMaxRefinesGrid(n_refinements);
		depthObj.SetAlphaGrid(sphcap_shrink);
		break;
	case eProjMeth::SimAnn:
		depthObj.SetAlphaSA(cooling_factor);
		depthObj.SetBetaSA(cap_size);
		depthObj.SetStartSA(start);
		break;
	case eProjMeth::CoordDesc:
		depthObj.SetLineSearchCD((eLineSearchCD)line_solver);
		break;
	case eProjMeth::CoordDescGC:
		depthObj.SetLineSearchCDGC((eLineSearchCDGC)line_solver);
		break;
	case eProjMeth::NelderMeadGC:
		depthObj.SetBetaNM(cap_size);
		depthObj.SetBoundNM(bound_gc);
		depthObj.SetStartNM(start);
		break;
	case eProjMeth::RandSimp:
		depthObj.SetAlphaRaSi(alpha_Dirichlet);
		break;
	default:
		break;
	}
	return 0;
	
}

void ACA(double *z, double *x, int notion, int solver, int NRandom, int n_refinements,
		double sphcap_shrink, double alpha_Dirichlet, double cooling_factor,
		double cap_size, int start, int line_solver, int bound_gc, int n, int d,
		int n_z, double *depths, double *best_directions, double *basis_py, int d_aca, int option){

	dyMatrixClass::cMatrix X(n ,d); // Fill data matrice with numpy array
	for(int i=0; i<n; i++){
		for(int j=0; j<d; j++){
			X[i][j] = x[j + i*d];
		}
	}

	dyMatrixClass::cMatrix basis(d, d_aca); // Fill the basis with identity matrix first and then null space
	for(int i=0; i<d; i++){
		for(int j=0; j<d_aca; j++){
			basis[i][j] = basis_py[j + i*d];
		}
	}
	
	cProjection Proj(X, n, d, NRandom);
	Proj.SetDepthNotion((eDepth)notion);
	Proj.SetMethod((eProjMeth)solver);
	Proj.SetOption(option);
	Proj.SetBasis(basis);
	Proj.SetD_ACA(d_aca);

	int setparam;
	setparam = SetDepthPars(Proj, n_refinements, sphcap_shrink, alpha_Dirichlet,
								cooling_factor, cap_size, start, line_solver, bound_gc,option);

	// std::vector<double> vec_depths;
	std::vector<double> vec_best_direction;
	
	for(int i=0; i<n_z; i++){
		depths[i] = Proj.Depth(z); // Calculate depth value
		z += d;
		vec_best_direction = Proj.BestDirection();
		for(int j=0; j<vec_best_direction.size(); j++){
			best_directions[j + i*d] = vec_best_direction[j]; // Keep only the best direction
		}
	}
}
};
