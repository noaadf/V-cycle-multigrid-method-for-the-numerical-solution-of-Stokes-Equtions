# include<cmath>
# include <cstdio>
# include<vector>
# include<Eigen/Sparse>
# include "utils.h"
# include <iostream> 

double F1(double x, double y){
    if(x<0 || x>1 || y<0 || y>1){
        throw std::out_of_range("Index out of range in F");
    }
    return -4*PI*PI*(2*cos(2*PI*x) - 1)*sin(2*PI*y) + x*x;
}

double F2(double x, double y){
    if(x<0 || x>1 || y<0 || y>1){
        throw std::out_of_range("Index out of range in F");
    }
    return 4*PI*PI*(2*cos(2*PI*y) - 1)*sin(2*PI*x);
}

std::pair<double, double> F(double x, double y){
    return {F1(x, y), F2(x, y)};
}

double u_0(double x, double y){
    return (1-cos(2*PI*x))*sin(2*PI*y);
}

double v_0(double x, double y){
    return -(1-cos(2*PI*y))*sin(2*PI*x);
}

double du_0(double x, double y){
    return (1-cos(2*PI*x))*2*PI*cos(2*PI*y);
}

double dv_0(double x, double y){
    return -(1-cos(2*PI*y))*2*PI*cos(2*PI*x);
}

double p_0(double x, double y){
    return x*x*x/3 - 1.0/12;
}

SparseM Init_A(int N){
    int n = 2*N*(N-1);
    int c = N*N;
    SparseM A(n, n);
    A.reserve(Eigen::VectorXi::Constant(n, 5));
    // u part
    // A_1 & -I & 0 & ... & 0
    // -I & A_2 & -I & ... & 0
    // 0 & -I & A_2 & ... & 0
    // ...
    // 0 & 0 & 0 & ... & A_1 
    for (int i=0; i<n/2; i++){
        int idx = i/(N-1);
        int idy = i%(N-1);
        if (idx == 0 || idx == N-1){
            A.insert(i, i) = 3*c;
            if (idy != 0){
                A.insert(i, i-1) = -c;
            }
            if (idy != N-2){
                A.insert(i, i+1) = -c;
            }
            if (idx == 0){
                A.insert(i, i+(N-1)) = -c;
            }
            if (idx == N-1){
                A.insert(i, i-(N-1)) = -c;
            }
        }
        else{
            int idy = i%(N-1);
            A.insert(i, i) = 4*c;
            if (idy != 0){
                A.insert(i, i-1) = -c;
            }
            if (idy != N-2){
                A.insert(i, i+1) = -c;
            }
            A.insert(i, i+(N-1)) = -c;
            A.insert(i, i-(N-1)) = -c;
        }
    }
    // v part
    // A_3 & -I & 0 & ... & 0
    // -I & A_3 & -I & ... & 0
    // 0 & -I & A_3 & ... & 0
    // ...
    // 0 & 0 & 0 & ... & A_3
    int bias = n/2;
    for (int i=0; i<n/2; i++){
        int idx = i/N;
        int idy = i%N;
        if (idy == 0 || idy == N-1){
            A.insert(i+bias, i+bias) = 3*c;
            if (idx != 0){
                A.insert(i+bias, i+bias-N) = -c;
            }
            if (idx != N-2){
                A.insert(i+bias, i+bias+N) = -c;
            }
            if (idy == 0){
                A.insert(i+bias, i+bias+1) = -c;
            }
            if (idy == N-1){
                A.insert(i+bias, i+bias-1) = -c;
            }
        }
        else{
            A.insert(i+bias, i+bias) = 4*c;
            if (idx != 0){
                A.insert(i+bias, i+bias-N) = -c;
            }
            if (idx != N-2){
                A.insert(i+bias, i+bias+N) = -c;
            }
            A.insert(i+bias, i+bias+1) = -c;
            A.insert(i+bias, i+bias-1) = -c;
        }
    }
    A.makeCompressed();
    return A;
}

SparseM Init_B(int N){
    int n = 2*N*(N-1);
    SparseM B(n, N*N);
    B.reserve(Eigen::VectorXi::Constant(n, 2));
    // u part
    // diag(H, H, H, ..., H)
    // H = -1 & 1 & 0 & ... & 0
    // 0 & -1 & 1 & ... & 0
    // 0 & 0 & -1 & ... & 0
    // ...
    // 0 & 0 & 0 & ... & 1
    for (int i=0; i<n/2; i++){
        int idx = i/(N-1);
        int idy = i%(N-1);
        B.insert(i, idx*N+idy) = -N;
        B.insert(i, idx*N+idy+1) = N;
    }
    // v part
    // B_3 & 0 & 0 & ... & 0
    // 0 & B_3 & 0 & ... & 0
    // 0 & 0 & B_3 & ... & 0
    // ...
    // 0 & 0 & 0 & ... & B_3
    int bias = n/2;
    for (int i=0; i<n/2; i++){
        B.insert(i+bias, i) = -N;
        B.insert(i+bias, i+N) = N;
    }
    B.makeCompressed();
    return B;
}


SparseM R_u(int N){
    //restriction operator R: 3*N**2-2N -> 3*(N/2)**2-N
    SparseM R(2*(N/2)*(N/2)-N, 2*N*N-2*N);
    R.reserve(Eigen::VectorXi::Constant(2*(N/2)*(N/2)-N, 6));
    // u part
    int n = N/2*(N/2-1);
    for (int i=0; i<n; i++){
        int idx = i/(N/2-1);
        int idy = i%(N/2-1);
        R.insert(i, 2*idx*(N-1)+(2*idy+1)) = 0.25;
        R.insert(i, (2*idx+1)*(N-1)+(2*idy+1)) = 0.25;
        R.insert(i, 2*idx*(N-1)+(2*idy)) = 0.125;
        R.insert(i, 2*idx*(N-1)+(2*idy+2)) = 0.125;
        R.insert(i, (2*idx+1)*(N-1)+(2*idy)) = 0.125;
        R.insert(i, (2*idx+1)*(N-1)+(2*idy+2)) = 0.125;
    }
    // v part
    int bias = N*(N-1);
    for (int i=0; i<n; i++){
        int idx = i/(N/2);
        int idy = i%(N/2);
        R.insert(i+n, (2*idx+1)*(N)+2*idy+bias) = 0.25;
        R.insert(i+n, (2*idx+1)*(N)+2*idy+1+bias) = 0.25;
        R.insert(i+n, (2*idx)*(N)+2*idy+bias) = 0.125;
        R.insert(i+n, (2*idx)*(N)+2*idy+1+bias) = 0.125;
        R.insert(i+n, (2*idx+2)*(N)+2*idy+bias) = 0.125;
        R.insert(i+n, (2*idx+2)*(N)+2*idy+1+bias) = 0.125;
    }
    R.makeCompressed();
    return R;
}

SparseM R_p(int N){
    //restriction operator R: N*N -> (N/2)*(N/2)
    SparseM R(N*N/4, N*N);
    for (int i=0; i<N*N/4; i++){
        int idx = i/(N/2);
        int idy = i%(N/2);
        R.insert(i, idx*2*N + 2*idy ) = 0.25;
        R.insert(i, idx*2*N + 2*idy + 1 ) = 0.25;
        R.insert(i, (2*idx+1)*N + 2*idy ) = 0.25;
        R.insert(i, (2*idx+1)*N + 2*idy + 1 ) = 0.25;
    }
    R.makeCompressed();
    return R;
}

Vec f(int N){
    int n = 2*N*(N-1);
    Vec f = Vec::Zero(n);
    for (int i=0; i<N; i++){
        for (int j=0; j<N-1; j++){
            int idx = i*(N-1)+j;
            f(idx) = F1((double(j+1))/N ,(i+0.5)/N) ;
            if(i == 0) f(idx) -= du_0(double(j+1)/N,0)*N;
            if(i == N-1) f(idx) += du_0(double(j+1)/N,1)*N;
        }
    }
    for (int i=0; i<N-1; i++){
        for (int j=0; j<N; j++){
            int idx = i*N+j+n/2;
            f(idx) = F2((j+0.5)/N ,(double(i+1))/N);
            if(j == 0) f(idx) -=dv_0(0,double(i+1)/N)*N;
            if(j == N-1 ) f(idx) +=dv_0(1,double(i+1)/N)*N;
        }
    }
    return f;
}

void CG_solver(SparseM &A, Vec &x, Vec f, int N, double tol){
    int n = 2*N*(N-1);
    Vec r = f - A*x;
    Vec p = r;
    int k = 0;
    double alpha, beta,rho,rho_old;
    double r_norm = r.norm();
    double f_dot_f = f.dot(f);
    double tqf = tol*tol*f_dot_f;
    rho = r.dot(r);
    while (rho > tqf && k<=1000){
        k++;
        if (k == 1){
            p = r;
        }
        else{
            beta = rho/rho_old;
            p = r + beta * p;
        }
        Vec w = A*p;
        alpha = rho/(p.dot(w));
        x = x + alpha*p;
        r = r - alpha*w;
        rho_old = rho;
        rho = r.dot(r);
    }
}

int inv_pow_method(SparseMC A, double x0){
    // generate a vector v randomly
    Vec v = Vec::Random(A.rows());
    v = v/v.norm();
    // iteration
    int k = 0;
    double lambda = 0;
    double lambda_old = 0;
    double tol = 1e-8;
    // solve (A - x0I)w = v
    SparseMC I(A.rows(), A.cols());
    I.setIdentity();
    SparseMC A_minus_x0I = A - x0*I;
    Eigen::SparseLU<SparseMC> solver;
    solver.compute(A_minus_x0I);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Matrix decomposition failed!" << std::endl;
        return -1;
    }
    while (k < 100){
        Vec w = solver.solve(v);
        lambda = w.norm();
        w = w/lambda;
        v = w;
        if (abs(lambda - lambda_old) < tol) break;
        lambda_old = lambda;
        k++;
    }
    std::cout<<"k: "<<k<<' '<<v.norm()<<std::endl;
    return 1/lambda + x0;
}

double cal_err(int N, Vec &u){
    double err = 0;
    // u part
    for (int i=0; i<N; i++){
        for (int j=0; j<N-1; j++){
            double x = (double(j+1))/N;
            double y = (i+0.5)/N;
            double u_real = u_0(x, y);
            int idx = i*(N-1)+j;
            err += (u_real - u[idx])*(u_real - u[idx]);
        }
    }
    // v part
    for (int i=0; i<N-1; i++){
        for (int j=0; j<N; j++){
            double x = (j+0.5)/N;
            double y = (double(i+1))/N;
            double v_real = v_0(x, y);
            int idx = i*N+j+N*(N-1);
            err += (v_real - u[idx])*(v_real - u[idx]);
        }
    }
    err = sqrt(err)/N;
    return err;
}

void GaussSeidel(const SparseM &A, Vec &u, const Vec &b) {
    const int rows = static_cast<int>(u.size());
    for (int j = 0; j < rows; j++) {
        double sum = 0;
        for (SparseM::InnerIterator it(A, j); it; ++it) {
            if (const long col = it.col(); col != j) {
                sum += it.value() * u[col];
            }
        }
        u[j] = (b[j] - sum) / A.coeff(j, j);
    }
}

void SGS(const SparseM &A, Vec &u, const Vec &b) {
    const int rows = static_cast<int>(u.size());
    for (int j = 0; j < rows; j++) {
        double sum = 0;
        for (SparseM::InnerIterator it(A, j); it; ++it) {
            if (const long col = it.col(); col != j) {
                sum += it.value() * u[col];
            }
        }
        u[j] = (b[j] - sum) / A.coeff(j, j);
    }
    for (int j = rows - 1; j >= 0; j--) {
        double sum = 0;
        for (SparseM::InnerIterator it(A, j); it; ++it) {
            if (const long col = it.col(); col != j) {
                sum += it.value() * u[col];
            }
        }
        u[j] = (b[j] - sum) / A.coeff(j, j);
    }
}

void redBlackGaussSeidel(const SparseM &A, Vec &u, const Vec &b,const int N) {
    const int rows = static_cast<int>(u.size());
    const int numU = N * (N - 1); // Size of u part
    const int numV = rows - numU; // Size of v part

    // Red and black node partitioning for u and v
    std::vector<int> redNodesU, blackNodesU;
    std::vector<int> redNodesV, blackNodesV;

    // Partition u (horizontal velocity)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N - 1; ++j) {
            int idx = i * (N - 1) + j;
            if ((i + j) % 2 == 0) {
                redNodesU.push_back(idx);
            } else {
                blackNodesU.push_back(idx);
            }
        }
    }

    // Partition v (vertical velocity)
    for (int i = 0; i < N - 1; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = numU + i * N + j;
            if ((i + j) % 2 == 0) {
                redNodesV.push_back(idx);
            } else {
                blackNodesV.push_back(idx);
            }
        }
    }

    auto updateNodes = [&](const std::vector<int> &nodes) {
        #pragma omp parallel for
        for (size_t k = 0; k < nodes.size(); ++k) {
            int j = nodes[k];
            double sum = 0;
            for (SparseM::InnerIterator it(A, j); it; ++it) {
                const long col = it.col();
                if (col != j) {
                    sum += it.value() * u[col];
                }
            }
            u[j] = (b[j] - sum) / A.coeff(j, j);
        }
    };

    // Red-black Gauss-Seidel updates
    updateNodes(redNodesU); // Update red nodes in u
    updateNodes(redNodesV); // Update red nodes in v
    updateNodes(blackNodesU); // Update black nodes in u
    updateNodes(blackNodesV); // Update black nodes in v
}

void update_grid(int N, Vec &u, Vec &p, double d, int idx, int idy, int j){
    int num_of_case = 4;
    int n = 2*N*(N-1);
    if(idx==0 || idx==N-1)num_of_case--;
    if(idy==0 || idy==N-1)num_of_case--;
    double rij = 0;
    int d1 = idx*(N-1)+idy;
    int d2 = idx*(N-1)+idy-1;
    int d3 = (idx)*N+idy;
    int d4 = (idx-1)*N+idy;
    if (idy != N-1)rij -= u[d1]*N;
    if (idy != 0)rij += u[d2]*N;
    if (idx != N-1)rij -= u[d3+n/2]*N;
    if (idx != 0)rij += u[d4+n/2]*N;
    rij -= d;
    double delta = rij/(N*num_of_case);
    if(idy!=0){u[d2] -= delta; p[j-1] -= rij/num_of_case;}
    if(idy!=N-1){u[d1] += delta; p[j+1] -= rij/num_of_case;}
    if(idx!=0){u[d4+n/2] -= delta; p[j-N] -= rij/num_of_case;}
    if(idx!=N-1){u[d3+n/2] += delta; p[j+N] -= rij/num_of_case;}
    p[j] += rij;
}
