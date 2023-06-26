#pragma once

#include "matrix.h"
#include "matrix_object.h"
#include "vector.h"
#include "matrix_operators.h"
#include "matrix_determinant_trace_norme.h"
#include "matrix_pivoting.h"
#include "matrix_inverse_transpose_adjugate.h"
#include "matrix_LDLT_decomposition.h"
#include "matrix_LU_decomposition.h"
#include "matrix_polar_decomposition_tools.h"
#include "matrix_polar_decomposition.h"

using namespace std;


namespace exanb {


  inline vector_ compute_eigenvalue(const matrix& A, bool debug_mod) {

    vector_ result(3);

    double p_1 = A.matrix_[1]*A.matrix_[1] + A.matrix_[2]*A.matrix_[2] + A.matrix_[5]*A.matrix_[5];

    if(p_1 == 0) {
      result.vecteur[0] = A.matrix_[0];
      result.vecteur[1] = A.matrix_[4];
      result.vecteur[2] = A.matrix_[8];
    }

    else {

      double q, p, p_2, r, phi=0;

      matrix B(4);

      q = A.trace()/3.;
      p_2 = (A.matrix_[0]-q)*(A.matrix_[0]-q) + (A.matrix_[4]-q)*(A.matrix_[4]-q) + (A.matrix_[8]-q)*(A.matrix_[8]-q) + 2*p_1;
      p = sqrt(p_2/6.);
      B = (A - q*identity_(3))/p;
      r = B.det()/2.;


      if(r<=-1)
          phi = M_PI/3.; 

      else if(r<=1)
          phi = acos(r)/3.;

      result.vecteur[0] = q+2*p*cos(phi);
      result.vecteur[1] = q+2*p*cos(phi+2*M_PI/3.);
      result.vecteur[2] = 3*q - result.vecteur[0] - result.vecteur[1];     
    }

    return result;
  };


  inline vector_ compute_eigenvector(const matrix& A, const vector_& eigenvalues, size_t value_index, bool debug_mod) {

    vector_ result(3);

    matrix I = identity_(3);

    matrix B(3);

    bool eigenvector_computed = false;

    for(size_t i=1; i<3; i++) {

      if(i != value_index) B = B*(A-eigenvalues.vecteur[i]*I);
    }

    size_t n_col = 0;

    while(!eigenvector_computed && n_col < 3) {

      for(size_t i=0; i<3; i++) {

        if(B.matrix_[n_col+i*3] != 0) eigenvector_computed = true;
      }

      if(eigenvector_computed) {
        for(size_t i=0; i<3; i++) result.vecteur[i] = B.matrix_[n_col+i*3];       
      }

      else n_col++;
    }

    if(!eigenvector_computed) {
      if(debug_mod) cout << "OUPSI ! pas de vecteur propre" << endl;

      result = zeros(3);
      return result;
    } 

    
    else {
      if(debug_mod) result.plot();
      return result;
    }
  };


  inline vector_ compute_eigenvector(const matrix& A, double eigenvalue, bool debug_mod) {

    vector_ result(3);
    matrix I = identity_(3);
    vector_ eigenvalues = compute_eigenvalue(A);
    matrix B(3);

    size_t n_col = 0;
    bool eigenvector_computed = false;

    for(size_t i=1; i<3; i++) B = B*(A-eigenvalues.vecteur[i]*I);


    while(!eigenvector_computed && n_col < 3) {

      for(size_t i=0; i<3; i++) {
        if(B.matrix_[n_col+i*3] != 0) eigenvector_computed = true; 
      }

      if(eigenvector_computed) {
        for(size_t i=0; i<3; i++) result.vecteur[i] = B.matrix_[n_col+i*3];           
      }


      else n_col++;
    }

    if(!eigenvector_computed) {
      if(debug_mod) cout << "OUPSI ! pas de vecteur propre" << endl;

      result = zeros(3);
      return result;
    } 
    
    else {
      if(debug_mod) result.plot();
      return result;
    }
  };

}
