#ifndef UTILS_H
#define UTILS_H

# include<cmath>
# include <cstdio>
# include<vector>
# include<chrono>
# include<Eigen/Sparse>
# include<numbers>
#include <omp.h>

using SparseM = Eigen::SparseMatrix<double, Eigen::RowMajor>;

using SparseMC = Eigen::SparseMatrix<double, Eigen::ColMajor>;

using Vec = Eigen::VectorXd;

const double PI = std::numbers::pi;

double F1(double x, double y);

double F2(double x, double y);

std::pair<double, double> F(double x, double y);

double u_0(double x, double y);

double v_0(double x, double y);

double du_0(double x, double y);

double dv_0(double x, double y);

double p_0(double x, double y);

SparseM Init_A(int N);

SparseM Init_B(int N);

SparseM R_u(int N);

SparseM R_p(int N);

Vec f(int N);

void CG_solver(SparseM &A, Vec &x, Vec f, const int N, double tol=1e-8);

int inv_pow_method(SparseMC A, double x0);

double cal_err(int N, Vec &u);

void GaussSeidel(const SparseM &A, Vec &u, const Vec &b);

void redBlackGaussSeidel(const SparseM &A, Vec &u, const Vec &b, int N);

void update_grid(int N, Vec &u, Vec &p, double d, int idx, int idy, int j, int num_of_case=4);

void SGS(const SparseM &A, Vec &u, const Vec &b);

#endif
