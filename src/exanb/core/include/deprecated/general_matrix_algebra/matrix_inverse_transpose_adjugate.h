#pragma once

////////////////////////////////////////////////////////////////////////////////
//                         "MATRICES DÉDUITES"
////////////////////////////////////////////////////////////////////////////////

#include "matrix.h"
#include "matrix_object.h"
#include "vector.h"
#include "matrix_operators.h"
#include "matrix_determinant_trace_norme.h"
#include "matrix_pivoting.h"
#include "matrix_3x3_eigen_decomposition.h"
#include "matrix_LDLT_decomposition.h"
#include "matrix_LU_decomposition.h"
#include "matrix_polar_decomposition_tools.h"
#include "matrix_polar_decomposition.h"

using namespace std;



namespace exanb {

  ////////////////////////////////////////////////////////////////////////////////
  //                              INVERSE
  ////////////////////////////////////////////////////////////////////////////////

  inline matrix matrix::inverse() const{ // pivot de Gauss

    assert(M==N);

    matrix mat_copy(M);
    size_t ided_line = -1; //nombres de lignes initialisées à 1
    size_t m, n;
    double pivot;
    size_t pivot_line_index;

    matrix pivot_matrix(M); //matrice pour les opérations de pivot

    matrix result = identity_(M);

    for (size_t i=0; i<M*N; i++)  mat_copy.matrix_[i] = matrix_[i];

    for (n=0; n<N; n++) {

      pivot = 0;
      pivot_line_index = 0;

      for(m=ided_line+1; m<M; m++) {
        if (abs(mat_copy.matrix_[n+m*N]) > abs(pivot)) {

            pivot = mat_copy.matrix_[n+m*N];
            pivot_line_index = m;
        }
      }

      if (abs(pivot) < 1e-16) {
        cout << "Matrice non inversible !" << endl;            
        assert(pivot > 1e-16);

        return result;
      }

      ided_line++;

      pivot_matrix = matrix_to_divide_a_line(pivot_line_index, pivot, M);

      matrix_update_front(mat_copy, result, pivot_matrix);

    if(pivot_line_index!=ided_line) {     //inversion des lignes

        pivot_matrix = line_inversion_matrix(pivot_line_index, ided_line, M);

        matrix_update_front(mat_copy, result, pivot_matrix);
      }

      pivot_matrix = mat_copy.matrix_to_zero_columns(ided_line, "inversion");

      matrix_update_front(mat_copy, result, pivot_matrix);
    }

    return result;
  };



  ////////////////////////////////////////////////////////////////////////////////
  //                              TRANSPOSÉE
  ////////////////////////////////////////////////////////////////////////////////

  inline matrix matrix::transpose() const { 

    matrix result(N, M);

    for(size_t m=0; m<N; m++) {
      for(size_t n=0; n<M; n++) {
        result.matrix_[n+m*M] = matrix_[m+n*N];    
      }
    }

    return result;
  };



  ////////////////////////////////////////////////////////////////////////////////
  //                           MATRICE ADJOINTE 
  ////////////////////////////////////////////////////////////////////////////////

  inline matrix matrix::adjugate() const {

    assert(M == N);

    matrix result(M);
    
    result = det()*inverse();
      
  //    matrix comatrix(M-1);
  //    size_t comatrix_avancement;

  //    for(size_t m=0; m<M; m++) {
  //        for(size_t n=0; n<N; n++) {
  //            
  //            comatrix_avancement = 0;

  //            for(size_t i=0; i<M; i++) {
  //                for(size_t j=0; j<M; j++) {

  //                    if((i != m) & (j !=n)) {

  //                        comatrix.matrix_[comatrix_avancement] = matrix_[j+i*M];
  //                        comatrix_avancement++;
  //                    }
  //                }   
  //            }

  //            result.matrix_[m+n*N] = pow(-1, m+n)*comatrix.det();
  //        }
  //    }

    return result;
  };

}
