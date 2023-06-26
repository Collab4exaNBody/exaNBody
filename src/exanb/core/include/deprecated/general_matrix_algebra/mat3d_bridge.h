#pragma once

#include <exanb/core/basic_types_def.h>

#include "exanb/general_matrix_algebra/matrix.h"
#include "exanb/general_matrix_algebra/matrix_object.h"
#include "exanb/general_matrix_algebra/vector.h"
#include "exanb/general_matrix_algebra/matrix_operators.h"
#include "exanb/general_matrix_algebra/matrix_determinant_trace_norme.h"
#include "exanb/general_matrix_algebra/matrix_pivoting.h"
#include "exanb/general_matrix_algebra/matrix_inverse_transpose_adjugate.h"
#include "exanb/general_matrix_algebra/matrix_3x3_eigen_decomposition.h"
#include "exanb/general_matrix_algebra/matrix_LDLT_decomposition.h"
#include "exanb/general_matrix_algebra/matrix_LU_decomposition.h"
#include "exanb/general_matrix_algebra/matrix_polar_decomposition_tools.h"
#include "exanb/general_matrix_algebra/matrix_polar_decomposition.h"



namespace exanb {

  inline matrix mat3d_to_matrix(const Mat3d& mat_3d) {

    matrix result(3);

    result.matrix_[0] = mat_3d.m11;
    result.matrix_[1] = mat_3d.m12;
    result.matrix_[2] = mat_3d.m13;

    result.matrix_[3] = mat_3d.m21;
    result.matrix_[4] = mat_3d.m22;
    result.matrix_[5] = mat_3d.m23;

    result.matrix_[6] = mat_3d.m31;
    result.matrix_[7] = mat_3d.m32;
    result.matrix_[8] = mat_3d.m33;

    return result;
  };


  inline Mat3d matrix_to_mat3d(const matrix& general_mat) {

    Mat3d result;

    bool is_nan = false;

    for(size_t i=0; i<9; i++) {if(isnan(general_mat.matrix_[i])) is_nan = true;}

    if(is_nan) result = make_identity_matrix();

    else {
      result.m11 = general_mat.matrix_[0];
      result.m12 = general_mat.matrix_[1];
      result.m13 = general_mat.matrix_[2];

      result.m21 = general_mat.matrix_[3];
      result.m22 = general_mat.matrix_[4];
      result.m23 = general_mat.matrix_[5];

      result.m31 = general_mat.matrix_[6];
      result.m32 = general_mat.matrix_[7];
      result.m33 = general_mat.matrix_[8];
    }

    return result;
  };


  inline void mat3d_polar_decomposition(Mat3d& mat_to_decompose, Mat3d& mat_ortho, Mat3d& mat_sym) {

    matrix I(3);
    I = identity_(3);

    matrix general_mat_to_decompose = mat3d_to_matrix(mat_to_decompose);

    matrix general_mat_sym(3);
    general_mat_sym = identity_(3);
 
    matrix general_mat_ortho(3);
    general_mat_ortho = identity_(3);
    
    general_mat_to_decompose.polar_decomposition(general_mat_ortho, general_mat_sym);

    if(((general_mat_ortho*general_mat_ortho.transpose()-I).square_Frobenius_norm() > 1e-6) || ((general_mat_sym-general_mat_sym.transpose()).square_Frobenius_norm() > 1e-6) || ((general_mat_ortho*general_mat_sym-general_mat_to_decompose).square_Frobenius_norm() > 4e-4*general_mat_to_decompose.square_Frobenius_norm())) {
      std::cout << "POLAR DECOMPOSITION FAILED : " << "mat_to_decompose-- " <<  mat_to_decompose << " | " << "error-- " << "R " << (general_mat_ortho*general_mat_ortho.transpose()-I).square_Frobenius_norm() << "   U " << (general_mat_sym-general_mat_sym.transpose()).square_Frobenius_norm() << "   RU " << (general_mat_ortho*general_mat_sym-general_mat_to_decompose).square_Frobenius_norm() << std::endl;
      general_mat_ortho = I;
      general_mat_sym = I;

      mat_to_decompose = make_identity_matrix();
    }
    
    mat_sym = matrix_to_mat3d(general_mat_sym);
    mat_ortho = matrix_to_mat3d(general_mat_ortho);

  };
};
