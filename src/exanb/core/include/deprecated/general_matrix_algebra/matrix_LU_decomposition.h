#pragma once

////////////////////////////////////////////////////////////////////////////////
//                            DÉCOMPOSTION LU
////////////////////////////////////////////////////////////////////////////////
#include "matrix.h"
#include "matrix_object.h"
#include "vector.h"
#include "matrix_operators.h"
#include "matrix_determinant_trace_norme.h"
#include "matrix_pivoting.h"
#include "matrix_inverse_transpose_adjugate.h"
#include "matrix_3x3_eigen_decomposition.h"
#include "matrix_LDLT_decomposition.h"
#include "matrix_polar_decomposition_tools.h"
#include "matrix_polar_decomposition.h"

using namespace std;



namespace exanb {

  struct vector_;
  struct matrix;

  ////////////////////////////////////////////////////////////////////////////////
  //                  DÉCOMPOSTION LU AVEC PIVOT PARTIEL
  ////////////////////////////////////////////////////////////////////////////////

  inline void matrix::partial_LU_decomposition(matrix& P, matrix& L, matrix& U) const {

    size_t m,n,max_line;
    double max;

    U = matrice();

    //Initialisations de L et P
    L = matrix_zeros(M);
    P = identity_(M);

    matrix pivot_matrix(M); //matrice support des transformations élémentaires

    for(n=0; n<M; n++) {
        
      max=0;
      max_line=n;

      for(m=n; m<M; m++) {
    
        if(abs(U.matrix_[n+m*N]) > abs(max)) {

          max = U.matrix_[n+m*N];
          max_line = m;
        }
      }

      if (max_line != n) {

        pivot_matrix = line_inversion_matrix(max_line, n, M);
        matrix_update_front(U, L, P, pivot_matrix);            
      }

      pivot_matrix = U.matrix_to_zero_columns(n, "LUdecomposition");

      L -= (pivot_matrix-identity_(M));   
      U = pivot_matrix*U; //une seule matrice à mettre à jour
    }

    L = L+identity_(M);    
      
  };




  ////////////////////////////////////////////////////////////////////////////////
  //                  DÉCOMPOSTION LU AVEC PIVOT COMPLET
  ////////////////////////////////////////////////////////////////////////////////

  inline void matrix::complete_LU_decomposition(matrix& P_l, matrix& P_c, matrix& L, matrix& U) const {

    size_t i,j,n,max_line,max_col;
    double max;

    U = matrice();

    //Initialisations de L et P
    L = matrix_zeros(M);
    P_l = identity_(M);
    P_c = identity_(M);

    matrix pivot_matrix(M); //matrice support des transformations élémentaires

    for(n=0; n<M; n++) {
        
      max=0;
      max_line=n;
      max_col=n;

      for(i=n; i<M; i++) {
        for(j=n; j<N; j++) {
    
          if(abs(U.matrix_[j+i*N]) > abs(max)) {

            max = U.matrix_[j+i*N];
            max_line = i;
            max_col = j;
          }
        }
      }

      if (max_line != n) {

        pivot_matrix = line_inversion_matrix(max_line, n, M);
        matrix_update_front(U, L, P_l, pivot_matrix);            
      }

      if (max_col != n) {

        pivot_matrix = line_inversion_matrix(max_line, n, M);
        matrix_update_back(U, L, P_c, pivot_matrix);            
      }

      pivot_matrix = U.matrix_to_zero_columns(n, "LUdecomposition");

      L -= (pivot_matrix-identity_(M));   
      U = pivot_matrix*U; //une seule matrice à mettre à jour
    }

    L = L+identity_(M);    
  };
}
