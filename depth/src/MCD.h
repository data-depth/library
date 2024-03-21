

double chisqr(double df, double quantile);
vector<int> split(int n);
vector<int> best(int p, vector<double>& all_det ,int rep);
void biased_cov(TDMatrix X, int n, int d, TDMatrix S);
void IndexUp(vector<int>& index, double* distTab);
void MeanCovUp(vector<int>& index, double* T,TDMatrix S, TDMatrix X, TDMatrix Xh,int n, int d, int h);
void DistanceUp(TDMatrix X, int n,  int d, double* distTab, double* T,TDMatrix S);
void cstep(vector<int>& index, double* distTab, double* T,TDMatrix S, TDMatrix X, TDMatrix Xh,int n, int d, int h);
void cstep_TSstart(vector<int>& index, double* distTab, double* T,TDMatrix S, TDMatrix X, TDMatrix Xh,int n, int d, int h);
void mcd_routine(vector<int>& index, double* distTab, double* T,TDMatrix S, TDMatrix X, TDMatrix Xh,int n, int d, int h);
void ExactUnivariateMcd(TDMatrix X, int n, int h, double* T, TDMatrix M);
void Mcd(TDMatrix X, int n, int d, int h, double* mat_MCD, double chisqr05, double chisqr0975, int mfull, int nstep, bool hiRegimeCompleteLastComp,int *seed, bool seeded);
