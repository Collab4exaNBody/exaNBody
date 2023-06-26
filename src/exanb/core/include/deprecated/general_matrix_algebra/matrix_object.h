#pragma once


#include "matrix.h"
#include "vector.h"
#include "matrix_operators.h"
#include "matrix_determinant_trace_norme.h"
#include "matrix_pivoting.h"
#include "matrix_inverse_transpose_adjugate.h"
#include "matrix_3x3_eigen_decomposition.h"
#include "matrix_LDLT_decomposition.h"
#include "matrix_LU_decomposition.h"
#include "matrix_polar_decomposition_tools.h"
#include "matrix_polar_decomposition.h"



using namespace std;



namespace exanb {


  //Constructeur et Destructeur
  inline matrix::matrix() {
    M=0;
    N=0;
  };


  inline matrix::matrix(size_t n_lin, size_t n_col) {

    M = n_lin, 0;
    N = n_col, 0;

    if((M>0)&&(N>0)) {
      matrix_ = new double[M*N];
      for(size_t i=0; i<M*N; i++) {matrix_[i] = 0;}
    }
    else {
      M=0;
      N=0;
    }
  };


  inline matrix::matrix(size_t n_lin) {

    M = n_lin;
    N = n_lin;
  
    if(n_lin!=0) {
      matrix_ = new double[M*N];
      for(size_t i=0; i<M*N; i++) {matrix_[i] = 0;}
    }
  };


  inline matrix::~matrix() {
    if ((M!=0) && (N!=0)) {
      M=0;
      N=0;
      delete[] matrix_;
    }
  };


  inline void matrix::plot() const {

    size_t m,n;

    for(m=0; m<M; m++) {

        cout << " [";

        for(n=0; n<N; n++) {

            cout << ' ' << matrix_[n+m*N];
        }
        cout << ']' << endl;
    }
    cout << endl;
  };


  inline matrix identity_(size_t M) {

    size_t m,n;

    matrix result(M);
    
    for(m=0; m<M; m++) {
      for(n=0; n<M; n++) {

        if(n==m) result.matrix_[n+m*M] = 1;

        else result.matrix_[n+m*M] = 0; 
      }
    }

    return result;
  };


  inline matrix matrix_zeros(size_t M) {

    size_t m,n;
    
    matrix result(M);
    
    for(m=0; m<M; m++) {
      for(n=0; n<M; n++) result.matrix_[n+m*M] = 0; 
    }

    return result;
  };


  inline matrix matrix_zeros(size_t M, size_t N) {

    matrix result(M, N);
    
    for(size_t m=0; m<M; m++) {
      for(size_t n=0; n<N; n++) result.matrix_[n+m*N] = 0; 

    }

    return result;
  };



  inline void matrix::operator=(double double_[]) {

    size_t m,n;

    for(m=0; m<M; m++) {
      for(n=0; n<N; n++) matrix_[n+m*N] = double_[n+m*N];
    }
  };


  inline void matrix::operator=(const matrix& Mat) {
      
    size_t m;
    
    if((M!=0) && (N!=0)) {
      delete[] matrix_;
    }

    M = Mat.M;
    N = Mat.N;
    
    if((M!=0) || (N!=0)) matrix_ = new double[M*N];
    for(m=0; m<M*N; m++) matrix_[m] = Mat.matrix_[m];
  };



  inline double& matrix::operator()(size_t m, size_t n) {
    return matrix_[n+m*N];
  };


  inline double matrix::operator()(size_t m, size_t n) const{
    return matrix_[n+m*N];
  };


  inline matrix matrix::matrice() const {

    assert(M != -1);
    size_t i;
    matrix result(M);

    for(i=0; i<M*N; i++) result.matrix_[i] = matrix_[i];

    return result;
  };

}
