#pragma once

#include "matrix.h"
#include "matrix_object.h"
#include "vector.h"
#include "matrix_operators.h"
#include "matrix_determinant_trace_norme.h"
#include "matrix_inverse_transpose_adjugate.h"
#include "matrix_3x3_eigen_decomposition.h"
#include "matrix_LDLT_decomposition.h"
#include "matrix_LU_decomposition.h"
#include "matrix_polar_decomposition_tools.h"
#include "matrix_polar_decomposition.h"

using namespace std;


namespace exanb {


  ////////////////////////////////////////////////////////////////////////////////
  //                          OPÉRATIONS PIVOTS
  ////////////////////////////////////////////////////////////////////////////////

  //pas une méthode sur une matrice pour éviter de le recalculer à chaque fois
  inline matrix line_inversion_matrix(size_t l1, size_t l2, size_t N) {

    matrix result(N);

    //construction de la matrice d'inversion de ligne
    for(size_t m=0; m<N; m++) {
      for(size_t n=0; n<N; n++) {

        if(((m==l1) & (n==l2)) || ((m==l2) && (n==l1)) || ((m==n) & (m!=l1) & (m!=l2)))
          result.matrix_[n+m*N] = 1;

        else  result.matrix_[n+m*N] = 0;                
      }
    }

    return result;
  }



  //mettre à zéro une colonne à l'exception de la diagonale
  inline matrix matrix::matrix_to_zero_columns(size_t column_to_compute_index, string inversion_or_LUdecomposition) const {

    size_t m;
    matrix result = identity_(M);

    for(m=0; m<M; m++) {

      if(((inversion_or_LUdecomposition == "inversion") & (m!=column_to_compute_index)) || ((inversion_or_LUdecomposition == "LUdecomposition") & (m>column_to_compute_index)))
        result.matrix_[column_to_compute_index+m*M] = -matrix_[column_to_compute_index + m*M]/matrix_[column_to_compute_index + column_to_compute_index*M];        
    }

    return result;
  };



  //matrice pour diviser une ligne par un réel
  inline matrix matrix_to_divide_a_line(size_t line_to_divide_index, double divider, size_t M) {

    matrix result = identity_(M);

    result.matrix_[line_to_divide_index+line_to_divide_index*M]=1./divider;

    return result;
  };



  //mises à jour des matrices par l'avant ou l'arrière
  inline void matrix_update_front(matrix& Mat1, matrix& Mat2, const matrix& operation_matrix) {

    Mat1 = operation_matrix*Mat1;
    Mat2 = operation_matrix*Mat2;
  };


  inline void matrix_update_front(matrix& Mat1, matrix& Mat2, matrix& Mat3, const matrix& operation_matrix) {

    Mat1 = operation_matrix*Mat1;
    Mat2 = operation_matrix*Mat2;
    Mat3 = operation_matrix*Mat3;
  }


  inline void matrix_update_back(matrix& Mat1, matrix& Mat2, const matrix& operation_matrix) {

    Mat1 = Mat1*operation_matrix;
    Mat2 = Mat2*operation_matrix;
  };


  inline void matrix_update_back(matrix& Mat1, matrix& Mat2, matrix& Mat3, const matrix& operation_matrix) {

    Mat1 = Mat1*operation_matrix;
    Mat2 = Mat2*operation_matrix;
    Mat3 = Mat3*operation_matrix;
  };



  inline vector_ matrix::column_extractor(size_t col_index) const {

    vector_ result(M);
    
    for(size_t m=0; m<M; m++) result.vecteur[m] = matrix_[col_index+m*N];
    
    return result;
  };


  inline void matrix::column_adder(const vector_& vec, size_t col_index) {

    size_t m;

    assert(M==vec.M);
    
    for(m=0; m<M; m++) matrix_[col_index+m*N] = vec.vecteur[m];

  };
}
