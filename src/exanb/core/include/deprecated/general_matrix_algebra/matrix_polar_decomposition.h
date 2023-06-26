#pragma once

////////////////////////////////////////////////////////////////////////////////
//                   DÉCOMPOSTION POLAIRE (uniquement pour les matrices 3x3)
////////////////////////////////////////////////////////////////////////////////

/*
Algorithme tiré de :

Higham, N.J., Noferini, V. An algorithm to compute the polar decomposition of a 3 × 3 matrix. Numer Algor 73, 349–369 (2016). https://doi.org/10.1007/s11075-016-0098-7
*/

#include "matrix.h"
#include "matrix_object.h"
#include "vector.h"
#include "matrix_operators.h"
#include "matrix_determinant_trace_norme.h"
#include "matrix_pivoting.h"
#include "matrix_inverse_transpose_adjugate.h"
#include "matrix_3x3_eigen_decomposition.h"
#include "matrix_LDLT_decomposition.h"
#include "matrix_LU_decomposition.h"
#include "matrix_polar_decomposition_tools.h"

using namespace std;


namespace exanb {


  //=============================================================== Algorithme 3.2
  inline void matrix::polar_decomposition_reference(matrix& Mat_ortho, matrix& Mat_semdef_pos, matrix& B, double& b, double norm, bool debug_mod) const {

    double d = det();
    double lambda_1;
    double content[4] = {0,0,0,1};


    if(d < 0) {

        if(debug_mod) cout << "d négatif" << endl;
        B = (-1)*B;
        d = -d;       
    }

    lambda_1 = B.compute_dominant_eigen_value(b, d);

    matrix B_s = lambda_1*identity_(4) - B;

    matrix P(4), L(4), D(4);

    B_s.diagonal_PLDLTPT_decomposition(P, L, D);
   
    vector_ e_4(4);

    e_4 = content;

    vector_ v(4);

    v = L.transpose().inverse()*e_4;

    v = P*v/v.norm();

    Mat_ortho = v.compute_Q_matrix();
   

    Mat_semdef_pos = Mat_ortho.transpose()*matrice()*norm;
     
    //"set the lower triangle equal to the upper triangle" 
    Mat_semdef_pos.matrix_[3] = Mat_semdef_pos.matrix_[1];
    Mat_semdef_pos.matrix_[6] = Mat_semdef_pos.matrix_[2];
    Mat_semdef_pos.matrix_[7] = Mat_semdef_pos.matrix_[5];
  };



  //=============================================================== Algorithme 3.5
  inline void matrix::polar_decomposition(matrix& Mat_ortho, matrix& Mat_semdef_pos, bool debug_mod) const {

    double frobenius_norm = sqrt(matrice().square_Frobenius_norm());
    double b;
    double log10_u22;
    size_t i, n_it;

    matrix A = matrice()/frobenius_norm;

    matrix B(4);
    
    B = A.compute_B_matrix();

    b = B.det(); 

    if(b < 1-1e-4) {
      if(debug_mod) 
        cout << "résolution standard" << endl;
      
      A.polar_decomposition_reference(Mat_ortho, Mat_semdef_pos, B, b, frobenius_norm);
    }
     
    else {
      if(debug_mod)  
        cout << "résolution non standard" << endl;

      matrix P_1(3), P_2(3), L(3), L4(4), U(3), D(4);

      A.complete_LU_decomposition(P_1, P_2, L, U);

      double d = P_1.det_permut()*P_2.det_permut()*L.det_diag()*U.det_diag(); //determinant de A avec la décomposition du turfu
      double lambda_1 = B.compute_dominant_eigen_value(b, d);

      
      if(d<0) B = B*(-1);

      matrix B_s = lambda_1*identity_(4) - B;
      
      log10_u22 = log10(abs(U.matrix_[1+1*U.N]));
      
      vector_ v(4);
        
      if(log10_u22 > -7.18) {
        
        double content[4] = {0,0,0,1};

        if(debug_mod) 
          cout << "log10 standard" << endl;
    
        n_it = int(15/(16.86+2*log10_u22))+1.;
        
        B_s.block_LDLT_decompostion(L4, D);
        
        vector_ e_4(4);

        e_4 = content;

        v = L4.transpose().inverse()*e_4;

        v = v/v.norm();
          
        for(i=0; i<n_it; i++) 
          B_s.inverse_iteration_step(v, lambda_1);
        
      }
       
      
      else {
        double lambda_min;
        double content[8] = {0,0,0,0,1,0,0,1};

        if(debug_mod) 
          cout << "log10 non standard" << endl;
        
        matrix e3_e4(4, 2);       

        e3_e4 = content;

        B_s.block_LDLT_decompostion(L4, D);
                 
        matrix V(4);
      
        V = L4.transpose().inverse()*e3_e4;
        
        //subspace algorithm iteration
        for(i=0; i<2; i++) {
        
            V.grahm_schmidt_algorithm();
            V = B_s*V;
        }           
        
        V.grahm_schmidt_algorithm();
 
        matrix B_p(4);
    
        B_p = V.transpose()*B_s*V;
        
        vector_ w(2); 
        
        lambda_min = 0.5*(B_p.matrix_[0]+B_p.matrix_[3]-sqrt(pow(B_p.matrix_[0]-B_p.matrix_[3],2) + 4*B_p.matrix_[1]*B_p.matrix_[2]));
        
        w.vecteur[0] = (lambda_min - B_p.matrix_[3])/B_p.matrix_[2];
        w.vecteur[1] = 1;
        
        v = V*w;
      }

      Mat_ortho = v.compute_Q_matrix();
      Mat_semdef_pos = Mat_ortho.transpose()*matrice();

    }   
  };
}
