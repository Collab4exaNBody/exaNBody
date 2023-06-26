#pragma once

////////////////////////////////////////////////////////////////////////////////
//                         DÉTERMINANT ET TRACE
////////////////////////////////////////////////////////////////////////////////

#include "matrix.h"
#include "matrix_object.h"
#include "vector.h"
#include "matrix_operators.h"
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
  //                              DÉTERMINANT
  ////////////////////////////////////////////////////////////////////////////////

  //déterminant matrice diagonale/triangulaire
  inline double matrix::det_diag() const {

    assert(M == N);

    double det = 1;
    
    for(size_t m=0; m<M; m++) det *= matrix_[m+m*M];           

    return det;
  };


  inline double det_diag(const matrix& Mat) {
      
    return Mat.det_diag();
  };




  //déterminant matrice de permutation
  inline double matrix::det_permut() const {

    assert(M == N);

    size_t m, n, count_inversions = 0;
    vector_ vec_123(M);
    
    for(m=0; m<M; m++) vec_123.vecteur[m] = m+1;
    
    vector_ vec_permut = matrice()*vec_123;
    
    for(m=0; m<M; m++) {
      for(n=m+1; n<M; n++) {
        
        if(vec_permut(m) > vec_permut(n)) count_inversions++;
      }
    }   
    
    return pow(-1, count_inversions);
  };


  inline double det_permut(const matrix& Mat) {
      
    return Mat.det_permut();
  };




  //calcul du déterminant à partir d'une décomposition PLU, voir si dans le cas 3x3 c'est pas plus rapide de multiplier les diagonales
  inline double matrix::det() const {

    assert(M == N);

    matrix P(M), U(M), L(M);

    partial_LU_decomposition(P, L, U);
    
    return L.det_diag()*U.det_diag()*P.det_permut();
  };



  inline double det(const matrix& Mat) {
      
    return Mat.det();
  };



  ////////////////////////////////////////////////////////////////////////////////
  //                                  TRACE
  ////////////////////////////////////////////////////////////////////////////////

  inline double matrix::trace() const {
      
    double result = 0;

    for(size_t m=0; m<M; m++) result += matrix_[m+m*M];

    return result;
  }


  ////////////////////////////////////////////////////////////////////////////////
  //                                NORMES
  ////////////////////////////////////////////////////////////////////////////////


  inline double matrix::max_matrix() const {

    double max = 0;
    double current_max;
    
    for(size_t i=0; i<M; i++) {
      for(size_t j=0; j<M; j++) {    

        current_max = abs(matrix_[j+i*N]);
        
        if(current_max > max) max = current_max;
      }
    }

    return max;
  };


  inline double matrix::square_Frobenius_norm() const {

    return (transpose()*matrice()).trace();
  };

}
