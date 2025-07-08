

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

int depth_approximation(double *z, double *x, int notion, int solver,
						int NRandom, int option, int n_refinements,
						double sphcap_shrink, double alpha_Dirichlet,
						double cooling_factor,double cap_size, int start,
						int line_solver, int bound_gc, int n, int d,
						int n_z, double *depths, double *depths_iter,
						double *directions, int *directions_card,
						double *best_directions){
	
	dyMatrixClass::cMatrix X(n ,d); // Fill data matrice with numpy array
	for(int i=0; i<n; i++){
		for(int j=0; j<d; j++){
			X[i][j] = x[j + i*d];
		}
	}

	cProjection Proj(X, n, d, NRandom);
	Proj.SetDepthNotion((eDepth)notion);
	Proj.SetMethod((eProjMeth)solver);
	Proj.SetD_ACA(d);

	int setparam;
	setparam = SetDepthPars(Proj, n_refinements, sphcap_shrink, alpha_Dirichlet,
								cooling_factor, cap_size, start, line_solver, bound_gc, option);

    // std::cout.precision(std::numeric_limits<double>::digits10 + 2);
	for(int i=0; i<n_z; i++){
		depths[i] = Proj.Depth(z); // Calculate depth value
		z += d;
        // std::cout << depths[i] << " ";
		
		if(option == 2 || option == 3 || option == 4){
			std::vector<double> vec_best_direction;
			vec_best_direction = Proj.BestDirection();
			for(int j=0; j<vec_best_direction.size(); j++){
				best_directions[j + i*d] = vec_best_direction[j]; // Keep only the best direction
			}

			if(option == 3 || option == 4){
				std::vector<double> vec_depths;
				vec_depths = Proj.Depths();
				for(int j=0; j<vec_depths.size(); j++){
					depths_iter[j] = vec_depths[j]; // Keep every depth value estimated
				}
				depths_iter += NRandom;

				if(option == 4){
					std::vector<std::vector<double>> vec_directions;
					std::vector<int> vec_directions_split;
					vec_directions = Proj.Directions();
					for(int j=0; j<vec_directions.size(); j++){
						for(int k=0; k<d; k++){
							directions[k + j*d] = vec_directions[j][k]; // Keep every direction used for projection
						}
					}
					directions += NRandom * d;

					vec_directions_split = Proj.DirectionsCard();
					for(int j=0; j<vec_directions_split.size(); j++){
						directions_card[j + i*NRandom] = vec_directions_split[j];
					}
				}
			}
		}
	}
	return 0;
}

int main() {
    std::cout << "Hello World!";
    return 0;
}


};
