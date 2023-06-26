#pragma once


////////////////////////////////////////////////////////////////////////////////
//                        OPÉRATIONS ÉLÉMENTAIRES
////////////////////////////////////////////////////////////////////////////////


#include "matrix.h"
#include "matrix_object.h"
#include "vector.h"
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

  ////////////////////////////////////////////////////////////////////////////////
  //                        ADDITION-SOUSTRACTION
  ////////////////////////////////////////////////////////////////////////////////


  inline matrix matrix::operator+(const matrix& Mat) const {

    size_t m,n;

    assert((M==Mat.M) & (N==Mat.N));

    matrix result(M, N);
    
    for(m=0; m<M; m++) {
      for(n=0; n<N; n++) result(m, n) = matrix_[n+m*N]+Mat.matrix_[n+m*N];
    }

    return result;
  };


  inline matrix matrix::operator-(const matrix& Mat) const {

    size_t m,n;

    assert((M==Mat.M) & (N==Mat.N));

    matrix result(M, N);
    
    for(m=0; m<M; m++) {
      for(n=0; n<N; n++) result(m, n) = matrix_[n+m*N]-Mat.matrix_[n+m*N];
    }

    return result;
  };


  inline void operator+=(matrix& Mat1, const matrix& Mat2) {
    Mat1 = Mat1+Mat2;
  }


  inline void operator-=(matrix& Mat1, const matrix& Mat2) {
    Mat1 = Mat1-Mat2;
  }



  ////////////////////////////////////////////////////////////////////////////////
  //                              PRODUITS
  ////////////////////////////////////////////////////////////////////////////////

  inline matrix matrix::operator*(double scal) const {

    size_t m,n;
    
    matrix result(M, N);
    
    for(m=0; m<M; m++) {
        for(n=0; n<N; n++) result(m, n) = matrix_[n+m*N]*scal;
    }

    return result;
  };


  inline matrix operator*(double scal, const matrix& Mat) {
    return Mat*scal;
  }


  inline vector_ matrix::operator*(const vector_& vec) const {

    size_t m,n;

    assert(N==vec.M);

    vector_ result(M);
    
    for(m=0; m<M; m++) {

      result.vecteur[m]=0;

      for(n=0; n<N; n++) result.vecteur[m] += matrix_[n+m*N]*vec.vecteur[n];

    }

    return result;
  };


  inline matrix matrix::operator*(const matrix& Mat) const {

    size_t m,n,o;

    assert(N==Mat.M);

    matrix result(M, Mat.N);

    for(m=0; m<M; m++) {

      for(n=0; n<Mat.N; n++) {

        result.matrix_[n+m*Mat.N] = 0;

        for(o=0; o<N; o++) result.matrix_[n+m*Mat.N] += matrix_[o+m*N]*Mat.matrix_[n+o*Mat.N];
      }
    }

    return result;
  };


  inline void operator*=(matrix& Mat1, const matrix& Mat2) {
    Mat1 = Mat1*Mat2;
  };


  inline matrix operator/(const matrix& Mat, double scal) {
    return Mat*(1/scal);
  };


  inline void operator/=(matrix& Mat1, double scal) {
    Mat1 = Mat1*(1/scal);
  };



  inline double matrix::double_contraction_product(const matrix& Mat) const {

    assert((M == Mat.M) & (N == Mat.N));

    size_t m,n;
    double result = 0;

    for(m=0; m<M; m++) {
        for(n=0; n<N; n++) result += matrix_[n+m*N]*Mat.matrix_[m+n*N];
    } 

    return result;
  };
}
