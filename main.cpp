# include<cmath>
# include <cstdio>
# include<vector>
# include<chrono>
#define EIGEN_NO_DEBUG
# include<Eigen/Sparse>
# include "utils.h"
# include <iostream>

std::vector<SparseM> Ru_list;
std::vector<SparseM> Rp_list;
std::vector<SparseM> Pu_list;
std::vector<SparseM> Pp_list;
std::vector<SparseM> A_list;
std::vector<SparseM> B_list;

void DGS_update(int N,int nu1,Vec &u, Vec &p, SparseM &A, SparseM &B, Vec &f, Vec &fp, bool Simplified = false){
    int n = 2*N*(N-1);
    int c = N*N;
    for(int i=0; i<nu1; i++){
        Vec r = f - B*p;
        // G-S pre-smoothing u
        if (!Simplified or N <= 8){
            GaussSeidel(A, u, r);
        }
        else{
            redBlackGaussSeidel(A, u, r, N);
        }
        
        if (N <= 8){
            for (int j = 0; j < c; j++) {
                int idx = j / N;
                int idy = j % N;
                update_grid(N, u, p, fp[j], idx, idy, j);
            }
        }
        else{ // Renew every mesh in parallel by dividing into 4 regions
            #pragma omp parallel for
            for (int region = 0; region < 8; region++) {
                int start_idx = region * (c / 8);
                int end_idx = (region == 7) ? c : (region + 1) * (c / 8);

                for (int j = start_idx; j < end_idx; j++) {
                    int idx = j / N;
                    int idy = j % N;
                    update_grid(N, u, p, fp[j], idx, idy, j);
                }
            }
        }
    }
    return;
}

void V_cycle(int N, SparseM &A, SparseM &B, Vec &u, Vec &p, Vec &fu, Vec &fp, int L, int nu1, int nu2,int L_max, bool Simplified = false){
    if(L==0){
        return;
    }
    // initialization
    int n = 2*N*(N-1);
    int c = N*N;
    int k = 0;
    SparseM Ru = Ru_list[L_max-L];
    SparseM Rp = Rp_list[L_max-L];
    SparseM Pu = Pu_list[L_max-L];
    SparseM Pp = Pp_list[L_max-L];

    Vec r = fu;
    DGS_update(N,nu1,u,p,A,B,fu,fp,Simplified);
    r = fu - A*u - B*p;
    // restriction
    Vec r2u = Ru*r;
    Vec r2p = Rp*(fp-B.transpose()*u);
    Vec u2 = Vec::Zero(N*(N/2-1));
    Vec p2 = Vec::Zero(c/4);
    if (A_list.size() <= L_max-L){
        if (Simplified){
            SparseM A2 = Init_A(N/2);
            SparseM B2 = Init_B(N/2);
            A_list.push_back(A2);
            B_list.push_back(B2);
        }
        else{
            SparseM A2 = Ru*A*Pu;
            SparseM B2 = Ru*B*Pp;
            A_list.push_back(A2);
            B_list.push_back(B2);
        }
    }
    SparseM A2 = A_list[L_max-L];
    SparseM B2 = B_list[L_max-L];
    // recursive
    V_cycle(N/2, A2, B2, u2, p2, r2u, r2p, L-1, nu1, nu2, L_max, Simplified);
    // prolongation
    u = u + Pu*u2;
    p = p + Pp*p2;
    DGS_update(N,nu2+L_max-L,u,p,A,B,fu,fp,Simplified);
    r = fu - A*u - B*p;
    k++;
    return;
}

std::pair<std::pair<Vec, Vec>, int> V_cycle_solve(int N, int L, int nu1, int nu2, bool Simplified = false){
    // initialization
    int n = 2*N*(N-1);
    int c = N*N;
    int k = 0;
    SparseM A = Init_A(N);
    SparseM B = Init_B(N);

    A_list.clear();
    B_list.clear();
    // save the restriction and prolongation operator
    Ru_list.clear();
    Rp_list.clear();
    Pu_list.clear();
    Pp_list.clear();
    int temp_n = N;
    for (int i=0; i<L; i++){
        SparseM Ru = R_u(temp_n);
        SparseM Rp = R_p(temp_n);
        Ru_list.push_back(Ru);
        Pu_list.push_back(Ru.transpose()*4);
        Rp_list.push_back(Rp);
        Pp_list.push_back(Rp.transpose()*4);
        temp_n = temp_n/2;
    }
    SparseM Ru = Ru_list[0];
    SparseM Rp = Rp_list[0];
    SparseM Pu = Pu_list[0];
    SparseM Pp = Pp_list[0];
    Vec u = Vec::Zero(n);
    Vec p = Vec::Zero(c);
    Vec f0 = f(N);
    Vec fp = Vec::Zero(c);
    Vec r = Vec::Zero(n);
    r = f0;
    double norm0 = r.norm();
    // V-cycle
    while( r.norm() > 1e-8*norm0 && k<=1000){
        DGS_update(N,nu1,u,p,A,B,f0,fp,Simplified);
        r = f0 - A*u - B*p;
        // restriction
        Vec r2u = Ru*r;
        Vec r2p = -Rp*(B.transpose()*u);
        Vec u2 = Vec::Zero(N*(N/2-1));
        Vec p2 = Vec::Zero(c/4);

        if (A_list.size() == 0){
            if (Simplified){
                SparseM A2 = Init_A(N/2);
                SparseM B2 = Init_B(N/2);
                A_list.push_back(A2);
                B_list.push_back(B2);
            }
            else{
                SparseM A2 = Ru*A*Pu;
                SparseM B2 = Ru*B*Pp;
                A_list.push_back(A2);
                B_list.push_back(B2);
            }
        }
        SparseM A2 = A_list[0];
        SparseM B2 = B_list[0];
        // recursive
        V_cycle(N/2, A2, B2, u2, p2, r2u, r2p, L-1, nu1, nu2, L, Simplified);
        // prolongation
        u = u + Pu*u2;
        p = p + Pp*p2;
        r = f0 - A*u - B*p;
        DGS_update(N,nu2,u,p,A,B,f0,fp,Simplified);
        r = f0 - A*u - B*p;
        k++;
    }
    return {{u, p}, k};
}

std::pair<std::pair<Vec,Vec>, int> Uzawa(int N){
    int n = 2*N*(N-1);
    int c = N*N;
    SparseM A = Init_A(N);
    SparseM B = Init_B(N);
    Vec u = Vec::Zero(n);
    Vec p = Vec::Zero(c);
    Vec f0 = f(N);
    Vec fp = Vec::Zero(c);
    double r0 = (f0 - A*u - B*p).norm();
    double r = r0;
    int k = 0;
    while(r > 1e-8*r0 && k <= 1000){
        // solve u
        CG_solver(A, u, f0 - B*p, N);
        p = p + B.transpose()*u;
        r = (f0 - A*u - B*p).norm();
        k++;
    }
    return {{u, p}, k};
}

void SGS_V_cycle(const SparseM &A, Vec &x, Vec b, int N, int nu1, int nu2,int iL){
    if (N == 2){
        for (int i=0; i<nu1+nu2; i++)
            GaussSeidel(A, x, b);
            // SGS(A, x, b);
        return;
    }
    int n = 2*N*(N-1);
    for (int i=0; i<nu1; i++)
        if (N <= 8){
            GaussSeidel(A, x, b);
        }
        else{
            redBlackGaussSeidel(A, x, b, N);
        }
        // SGS(A, x, b);
    Vec r = b - A*x;
    // restriction
    Vec r2 = Ru_list[iL]*r;
    Vec x2 = Vec::Zero(N*(N/2-1));
    if (A_list.size() <= iL+1){
        A_list.emplace_back(Init_A(N/2));
    }
    SparseM A2 = A_list[iL+1];
    // recursive
    SGS_V_cycle(A2, x2, r2, N/2, nu1, nu2, iL+1);
    // prolongation
    x = x + Pu_list[iL]*x2;
    for (int i=0; i<nu2+iL; i++)
        // GaussSeidel(A, x, b);
        // SGS(A, x, b);
        if(N <= 8){
            GaussSeidel(A, x, b);
        }
        else{
            redBlackGaussSeidel(A, x, b, N);
        }
    return;
}

int PCG_solver(SparseM &A, SparseM &B_T, Vec &x, Vec b, int N, double tau, int nu1, int nu2, double eps=1e-8){
    int n = 2*N*(N-1);
    Vec r = b - A*x;
    double rho = r.dot(r);
    double rho_old = rho;
    int k = 0;
    Vec p = Eigen::VectorXd::Zero(n);
    Vec w = Eigen::VectorXd::Zero(n);
    Vec z = Eigen::VectorXd::Zero(n);
    double alpha = 0, beta = 0;
    while(sqrt(rho) > tau * (B_T*x).norm() && k <= 100 && sqrt(rho) > eps*b.norm()){
        k++;
        double r0 = r.norm();
        int k1 = 0;
        while((A*z-r).norm() > 1e-3 * r0 && k1<=3){
            SGS_V_cycle(A, z, r, N, nu1, nu2, 0);
            k1++;
        }
        if(k == 1){
            p = z; 
            rho = r.dot(z);
        }
        else{
            rho_old = rho;
            rho = r.dot(z);
            beta = rho / rho_old;
            p = z + beta * p;
        }
        w = A * p;
        alpha = rho / p.dot(w);
        x = x + alpha * p;
        r = r - alpha * w;
    }
    return k;
}

std::pair<std::pair<Vec,Vec>, std::pair<int,std::vector<int>>> I_Uzawa(int N, double alpha, double tau, int nu1, int nu2){
    A_list.clear();
    Ru_list.clear();
    Pu_list.clear();
    int temp_n = N, L = log(N)/log(2);
    for (int i=0; i<L; i++){
        SparseM Ru = R_u(temp_n);
        Ru_list.push_back(Ru);
        Pu_list.push_back(Ru.transpose()*4);
        temp_n = temp_n/2;
    }
    SparseM A = Init_A(N);
    SparseM B = Init_B(N);
    A_list.push_back(A);
    int n = 2*N*(N-1);
    int c = N*N;
    Vec u = Vec::Zero(n);
    Vec p = Vec::Zero(c);
    Vec f0 = f(N);
    int k = 0;
    double r0 = (f0 - A*u - B*p).norm();
    double r = r0;
    SparseM B_T = B.transpose();
    std::vector<int> pcg_itrs; 
    while(r > 1e-8*r0 && k <= 1000){
        // solve u
        int itr = PCG_solver(A, B_T, u, f0 - B*p, N, tau, nu1, nu2);
        pcg_itrs.push_back(itr);
        // solve p
        p = p + alpha * B_T * u;
        r = (f0 - A*u - B*p).norm();
        k++;
    }
    return {{u, p}, {k, pcg_itrs}};
}

int main(int argc, char* argv[])
{
    if (argc == 2 && (std::string(argv[1]) == "DGS" || std::string(argv[1]) == "UID" || std::string(argv[1]) == "IUID"))
        std::cout<<argv[0]<<' '<<"starts."<<std::endl<<"Using "<<argv[1]<<" method:"<<std::endl;
        // int i = 0;
    else {std::cout<<"Usage: " << argv[0] << " <method> [DSG,UID,IUID]" << std::endl; }
    if (std::string(argv[1]) == "DGS"){
        int numbers[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
        double error[8];
        for(int i=0; i<7; i++){
            int N = numbers[i];
            int L = i+6;
            auto start = std::chrono::high_resolution_clock::now();
            auto [u, k] = V_cycle_solve(N, L, 3, 1, true);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            printf("N: %d, k: %d, time: %.10f\n", N, k, elapsed.count());
            // calculate error
            double err = cal_err(N, u.first);
            error[i] = err;
            printf("N: %d, error: %.10f\n", N, err);
        }
    }
    if (std::string(argv[1]) == "UID"){
        if(argc != 2){ // check the eigenvalue
            int n = 32;
            SparseM A0 = Init_A(n);
            SparseM B0 = Init_B(n);
            SparseMC A = A0;
            SparseMC B = B0;
            Eigen::SparseLU<SparseMC> solver;
            solver.compute(A);

            if (solver.info() != Eigen::Success) {
                std::cerr << "Matrix decomposition failed!" << std::endl;
                return -1;
            }
            // 构造单位矩阵
            SparseMC I(A.rows(), A.cols());
            I.setIdentity();
            // 计算逆矩阵
            SparseMC A_inv = solver.solve(I);
            SparseMC C = B.transpose()*A_inv*B;
            // 反幂法求B^T A^{-1} B 的特征根
            std::cout<<inv_pow_method(C, 0.1)<<std::endl;
            return 0;
        }
        int numbers[] = {64, 128, 256, 512};
        double error[4];
        for (int i = 0; i < 4; i++){
            int N = numbers[i];
            auto start = std::chrono::high_resolution_clock::now();
            auto [u, k] = Uzawa(N);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            printf("N: %d, k: %d, time: %.10f\n", N, k, elapsed.count());
            // calculate error
            double err = cal_err(N, u.first);
            error[i] = err;
            printf("N: %d, error: %.10f\n", N, err);
        }

    }
    if(std::string(argv[1]) == "IUID"){
        int numbers[] = {64, 128, 256, 512, 1024, 2048, 4096};
        double error[7];
        double alpha = 1.0, tao = 1e-3;
        int nu1 = 1, nu2 = 1;
        for (int i = 0; i < 7; i++){
            int N = numbers[i];
            auto start = std::chrono::high_resolution_clock::now();
            auto [u, ks] = I_Uzawa(N, alpha, tao, nu1, nu2);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            printf("N: %d, k: %d, time: %.10f\n", N, ks.first, elapsed.count());
            printf("PCG iterations: ");
            for (int k : ks.second){
                printf("%d ", k);
            }
            printf("\n");
            // calculate error
            double err = cal_err(N, u.first);
            error[i] = err;
            printf("N: %d, error: %.10f\n", N, err);
        }
    }
    return 0;
}